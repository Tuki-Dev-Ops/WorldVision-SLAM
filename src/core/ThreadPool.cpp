#include "wme/core/ThreadPool.hpp"

#include <algorithm>
#include <exception>
#include <limits>

namespace wme {
namespace {

// 현재 스레드가 어느 풀의 워커인지. parallelFor 의 자기재귀 데드락 방지에 쓴다.
thread_local const void* t_owning_pool = nullptr;

// parallelFor 한 번의 동기화 상태. 호출 스레드 스택에 놓인다.
// 워커가 이 객체를 건드리는 마지막 시점은 remaining 의 감소이고, 깨우기는
// 풀이 소유한 뮤텍스/조건변수로 하므로 스택 해제와 경쟁하지 않는다.
struct ParallelSync {
    const std::function<void(std::size_t, std::size_t)>* body{nullptr};
    std::atomic<std::size_t> remaining{0};

    // 예외는 드물다. 여기서만 락을 잡으므로 정상 경로에는 비용이 없다.
    std::mutex         err_mutex;
    std::exception_ptr err;
    std::size_t        err_lo{(std::numeric_limits<std::size_t>::max)()};

    void record(std::size_t lo) {
        std::lock_guard lk(err_mutex);
        if (err == nullptr || lo < err_lo) {
            err    = std::current_exception();
            err_lo = lo;
        }
    }
};

}  // namespace

ThreadPool::ThreadPool(unsigned workers, std::string name) : name_(std::move(name)) {
    if (workers == 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        workers = (hw > 1) ? hw - 1 : 1;
    }
    queues_.reserve(workers);
    for (unsigned i = 0; i < workers; ++i) queues_.push_back(std::make_unique<Queue>());

    workers_.reserve(workers);
    for (unsigned i = 0; i < workers; ++i) {
        workers_.emplace_back([this, i] {
            t_owning_pool = this;
            workerLoop(i);
            t_owning_pool = nullptr;
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard lk(cv_mutex_);
        stopping_.store(true, std::memory_order_release);
    }
    work_cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
}

ThreadPool& ThreadPool::global() {
    static ThreadPool pool{};
    return pool;
}

void ThreadPool::enqueue(std::function<void()> job) {
    if (workers_.empty()) {   // 워커가 없으면 즉시 실행 (단일 스레드 디버깅 경로)
        job();
        return;
    }
    const std::size_t idx = round_robin_.fetch_add(1, std::memory_order_relaxed) % queues_.size();

    in_flight_.fetch_add(1, std::memory_order_acq_rel);
    {
        std::lock_guard lk(queues_[idx]->mutex);
        queues_[idx]->push(std::move(job));
        // 큐에 들어간 뒤에 카운터를 올려야 워커가 헛돌지 않는다
        queued_.fetch_add(1, std::memory_order_release);
    }
    work_cv_.notify_one();
}

bool ThreadPool::tryPopLocal(unsigned index, std::function<void()>& out) {
    auto& q = *queues_[index];
    std::lock_guard lk(q.mutex);
    if (q.empty()) return false;
    q.popFront(out);
    queued_.fetch_sub(1, std::memory_order_acq_rel);
    return true;
}

bool ThreadPool::trySteal(unsigned index, std::function<void()>& out) {
    const std::size_t n = queues_.size();
    for (std::size_t k = 1; k < n; ++k) {
        auto& q = *queues_[(index + k) % n];
        std::lock_guard lk(q.mutex);
        if (q.empty()) continue;
        q.popBack(out);   // 반대쪽 끝에서 훔쳐 소유 워커와의 경합을 줄인다
        queued_.fetch_sub(1, std::memory_order_acq_rel);
        return true;
    }
    return false;
}

void ThreadPool::workerLoop(unsigned index) {
    std::function<void()> job;
    while (true) {
        if (tryPopLocal(index, job) || trySteal(index, job)) {
            job();
            job = nullptr;
            if (in_flight_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard lk(cv_mutex_);
                idle_cv_.notify_all();
            }
            continue;
        }
        std::unique_lock lk(cv_mutex_);
        // 대기 조건은 "대기 중인 작업"이지 "실행 중인 작업"이 아니다.
        // in_flight_ 로 기다리면 다른 워커가 실행하는 동안 계속 깨어나 헛돈다.
        work_cv_.wait(lk, [this] {
            return stopping_.load(std::memory_order_acquire) ||
                   queued_.load(std::memory_order_acquire) > 0;
        });
        if (stopping_.load(std::memory_order_acquire) &&
            queued_.load(std::memory_order_acquire) == 0) {
            return;
        }
    }
}

void ThreadPool::parallelFor(std::size_t begin, std::size_t end,
                             const std::function<void(std::size_t, std::size_t)>& body,
                             std::size_t min_chunk) {
    if (begin >= end) return;

    // 워커 스레드에서 호출되면 분할하지 않는다. 여기서 future 를 기다리면
    // 그 워커가 점유된 채 블록되어 풀 전체가 굶을 수 있다.
    if (workers_.empty() || t_owning_pool == this) {
        body(begin, end);
        return;
    }

    const std::size_t total  = end - begin;
    const std::size_t slices = workers_.size() + 1;   // 호출 스레드도 한 몫 처리
    std::size_t chunk = (total + slices - 1) / slices;
    chunk = std::max(chunk, std::max<std::size_t>(min_chunk, 1));

    if (chunk >= total) {   // 분할 이득이 없다
        body(begin, end);
        return;
    }

    // future 를 쓰지 않는다. submit 한 번마다 packaged_task 의 공유 상태와
    // shared_ptr 제어블록이 생겨 덩어리마다 두 번씩 할당됐다. 남는 것은
    // "몇 개 남았나" 뿐이므로 스택 카운터 하나로 충분하다.
    ParallelSync sync;
    sync.body = &body;

    // 아래 루프가 도는 횟수. 조건이 (c + chunk < end) 이므로 k 번째 덩어리는
    // (k+1)*chunk < total 일 때만 던져진다 -> 횟수 = floor((total-1)/chunk).
    // 카운터는 반드시 던지기 *전에* 확정돼야 한다. 먼저 던지고 세면 워커가
    // 그 사이에 끝나면서 0 이 아닌 값을 지나쳐 버린다.
    const std::size_t posted = (total - 1) / chunk;
    sync.remaining.store(posted, std::memory_order_relaxed);

    ParallelSync* const s = &sync;
    std::size_t cursor = begin;
    for (; cursor + chunk < end; cursor += chunk) {
        const std::size_t lo = cursor, hi = cursor + chunk;
        // 캡처는 포인터 2개 + size_t 2개 = 32 B. std::function 의 내부 버퍼에
        // 들어가므로 클로저 자체로는 할당이 생기지 않는다.
        post([this, s, lo, hi] {
            try {
                (*s->body)(lo, hi);
            } catch (...) {
                s->record(lo);
            }
            if (s->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                // 깨우기는 풀 소유 객체로만 한다. 여기서 s 를 다시 읽으면
                // 호출 스레드가 이미 스택을 걷어낸 뒤일 수 있다.
                std::lock_guard lk(parallel_mutex_);
                parallel_cv_.notify_all();
            }
        });
    }

    // 마지막 덩어리는 호출 스레드가 직접. 여기서 던져도 나머지 덩어리가
    // body 를 참조하고 있으므로 먼저 전부 끝난 것을 확인한 뒤에 되던진다.
    try {
        body(cursor, end);
    } catch (...) {
        sync.record(cursor);
    }

    if (posted > 0) {
        std::unique_lock lk(parallel_mutex_);
        parallel_cv_.wait(lk, [&sync] {
            return sync.remaining.load(std::memory_order_acquire) == 0;
        });
    }

    if (sync.err) std::rethrow_exception(sync.err);
}

std::size_t ThreadPool::pending() const {
    return in_flight_.load(std::memory_order_acquire);
}

void ThreadPool::waitIdle() {
    std::unique_lock lk(cv_mutex_);
    idle_cv_.wait(lk, [this] { return in_flight_.load(std::memory_order_acquire) == 0; });
}

}  // namespace wme
