#pragma once

// 고정 크기 워커 풀. 워커별 로컬 큐 + 작업 훔치기로 경합을 줄인다.
// 엔진 전체가 단일 인스턴스를 공유한다.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace wme {

class ThreadPool {
public:
    // workers = 0 이면 하드웨어 동시성 - 1
    explicit ThreadPool(unsigned workers = 0, std::string name = "wme");
    ~ThreadPool();

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using R = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<R()>>(
            [fn = std::forward<F>(f), ... a = std::forward<Args>(args)]() mutable -> R {
                return fn(std::move(a)...);
            });
        auto fut = task->get_future();
        enqueue([task] { (*task)(); });
        return fut;
    }

    // 반환값이 필요 없는 경로. future 할당을 피한다.
    void post(std::function<void()> job) { enqueue(std::move(job)); }

    // [begin, end) 를 워커 수만큼 분할해 실행하고 전부 끝날 때까지 대기.
    // 호출 스레드도 한 덩어리를 처리한다.
    // 워커 스레드 안에서 호출하면 분할 없이 인라인 실행한다 (자기재귀 데드락 방지).
    // body 가 던진 예외는 모든 덩어리가 끝난 뒤 호출 스레드로 다시 던진다.
    // 여러 덩어리가 던지면 가장 앞선 구간의 예외를 쓴다.
    // 정상 상태에서 힙 할당 없음 (작업 큐가 링 버퍼, 동기화 객체가 스택).
    void parallelFor(std::size_t begin, std::size_t end,
                     const std::function<void(std::size_t, std::size_t)>& body,
                     std::size_t min_chunk = 1);

    [[nodiscard]] unsigned workerCount() const { return static_cast<unsigned>(workers_.size()); }
    [[nodiscard]] std::size_t pending() const;

    void waitIdle();

    // 프로세스 전역 기본 풀
    static ThreadPool& global();

private:
    void enqueue(std::function<void()> job);
    void workerLoop(unsigned index);
    bool tryPopLocal(unsigned index, std::function<void()>& out);
    bool trySteal(unsigned index, std::function<void()>& out);

    // 큐 하나의 초기 용량. 정상 상태에서 링이 자라지 않을 만큼만 잡는다.
    static constexpr std::size_t kInitialQueueCapacity = 256;

    // 워커별 작업 큐. std::deque 는 원소가 커서(std::function 64 B) 블록당 1개만
    // 담기고, push 마다 노드를 새로 잡는다 - parallelFor 한 번에 덩어리 수만큼
    // 할당이 생겼다. 양끝 접근만 필요하므로 링 버퍼로 바꿔 정상 상태 할당을 없앤다.
    // 용량은 2의 거듭제곱으로 유지해 나머지 연산을 비트 마스크로 처리한다.
    struct Queue {
        mutable std::mutex                 mutex;
        std::vector<std::function<void()>> ring;
        std::size_t                        head{0};    // 다음에 꺼낼 자리
        std::size_t                        count{0};

        Queue() : ring(kInitialQueueCapacity) {}

        [[nodiscard]] bool empty() const { return count == 0; }

        void push(std::function<void()>&& job) {
            if (count == ring.size()) grow();
            ring[(head + count) & (ring.size() - 1)] = std::move(job);
            ++count;
        }
        void popFront(std::function<void()>& out) {
            out = std::move(ring[head]);
            ring[head] = nullptr;
            head = (head + 1) & (ring.size() - 1);
            --count;
        }
        void popBack(std::function<void()>& out) {
            const std::size_t i = (head + count - 1) & (ring.size() - 1);
            out = std::move(ring[i]);
            ring[i] = nullptr;
            --count;
        }

    private:
        void grow() {
            std::vector<std::function<void()>> next(ring.size() * 2);
            for (std::size_t k = 0; k < count; ++k) {
                next[k] = std::move(ring[(head + k) & (ring.size() - 1)]);
            }
            ring.swap(next);
            head = 0;
        }
    };

    std::string                          name_;
    std::vector<std::thread>             workers_;
    std::vector<std::unique_ptr<Queue>>  queues_;
    std::atomic<std::size_t>             round_robin_{0};
    std::atomic<std::size_t>             queued_{0};      // 대기 중 작업 수
    std::atomic<std::size_t>             in_flight_{0};   // 대기 + 실행 중 작업 수
    std::atomic<bool>                    stopping_{false};

    mutable std::mutex        cv_mutex_;
    std::condition_variable   work_cv_;
    std::condition_variable   idle_cv_;

    // parallelFor 전용 대기점. 풀 수명 동안 살아 있으므로 호출 스택에 놓인
    // 동기화 카운터가 먼저 사라져도 워커가 죽은 메모리를 건드리지 않는다.
    // work_cv_ 와 뮤텍스를 공유하면 워커 루프와 매 덩어리마다 경합한다.
    mutable std::mutex        parallel_mutex_;
    std::condition_variable   parallel_cv_;
};

}  // namespace wme
