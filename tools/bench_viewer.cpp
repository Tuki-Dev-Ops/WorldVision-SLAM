// WorldVision-SLAM 벤치마크 뷰어 (네이티브 실행 파일).
//
// 왼쪽에 고전 기술자 파이프라인(ORB+PnP), 오른쪽에 WME 를 놓고 같은 프레임을
// 같은 속도로 재생하면서 궤적과 지표를 나란히 보여 준다.
//
// **지표를 여기서 계산하지 않는다.** ATE/RPE/속도는 전부
// results/bench/viewer.tsv 에서 읽는다. 그 파일은 python/tools/bench_run.py 가
// 계산한 값을 그대로 옮긴 것이다. 뷰어가 다시 계산하면 화면의 숫자와 문서의
// 숫자가 갈릴 수 있고, 그러면 어느 쪽이 맞는지 아무도 모른다. 매니페스트가
// 없으면 실행을 거부한다 - 0 을 그리고 마는 것이 제일 나쁘다.
//
// 궤적 정렬(Kabsch)만 여기서 한다. 그것은 **표시용** 이다: 추정 궤적은 자기
// 좌표계에 있으므로 정렬 없이 겹쳐 그리면 아무 것도 비교되지 않는다. 그리고
// 화면에 그 사실을 적어 둔다.
//
// 의존성은 OpenCV highgui/imgproc/imgcodecs 뿐이다. 새 라이브러리를 들이지
// 않는다 - 뷰어 하나 때문에 빌드가 무거워지면 아무도 안 켠다.
//
// 조작:  SPACE 재생/정지 · ← → 한 프레임 · N/P 시퀀스 · R 처음으로
//        F 자동맞춤 · S 스크린샷 · Q/ESC 종료

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// --- 색 (BGR). 어두운 배경에 두 계열을 확실히 갈라 놓는다 -------------------
const cv::Scalar kBg      (24, 22, 20);
const cv::Scalar kPanel   (38, 34, 31);
const cv::Scalar kPanelHi (52, 47, 43);
const cv::Scalar kInk     (238, 238, 240);
const cv::Scalar kInkDim  (150, 148, 145);
const cv::Scalar kInkFaint(92, 88, 85);
const cv::Scalar kLeft    (86, 140, 232);    // 주황빛 - 고전
const cv::Scalar kRight   (176, 200, 92);    // 청록빛 - WME
const cv::Scalar kGt      (140, 140, 140);
const cv::Scalar kWin     (120, 210, 120);
const cv::Scalar kLose    (110, 110, 220);

struct Pose {
    double t{0.0};
    Eigen::Vector3d p{Eigen::Vector3d::Zero()};
};

struct Run {
    std::string system, label, kind, status, traj_file;
    double ate_cm{std::nan("")}, rpe_mm{std::nan("")}, ms{std::nan("")};
    int    frames{0};
    std::vector<Pose> traj;      // 원본
    std::vector<cv::Point2f> xy; // 정렬 + 투영 후 화면 좌표계 이전의 2D
};

struct Seq {
    std::string name, dataset, dir, gt_file;
    double identity_ate_cm{std::nan("")};
    std::vector<Pose> gt;
    std::vector<cv::Point2f> gt_xy;
    std::vector<std::pair<double, std::string>> rgb;   // (stamp, path)
    std::map<std::string, Run> runs;
    bool loaded{false};
};

// --- 파싱 -------------------------------------------------------------------

std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream ss(s);
    while (std::getline(ss, cur, d)) out.push_back(cur);
    return out;
}

double toD(const std::string& s) {
    try { return std::stod(s); } catch (...) { return std::nan(""); }
}

std::vector<Pose> readTum(const std::string& path) {
    std::vector<Pose> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Pose p;
        double qx, qy, qz, qw;
        if (ss >> p.t >> p.p.x() >> p.p.y() >> p.p.z() >> qx >> qy >> qz >> qw) {
            out.push_back(p);
        }
    }
    return out;
}

std::vector<std::pair<double, std::string>> readIndex(const std::string& dir,
                                                      const std::string& name) {
    std::vector<std::pair<double, std::string>> out;
    std::ifstream f(dir + "/" + name);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double t;
        std::string rel;
        if (ss >> t >> rel) out.emplace_back(t, dir + "/" + rel);
    }
    return out;
}

// --- 표시용 정렬 ------------------------------------------------------------
//
// Kabsch: 추정 궤적을 정답 좌표계로 회전/평행이동만으로 맞춘다. 스케일은
// 건드리지 않는다 - 스케일까지 맞추면 스케일 드리프트가 화면에서 사라진다.
struct Rigid {
    Eigen::Matrix3d R{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d t{Eigen::Vector3d::Zero()};
};

Rigid kabsch(const std::vector<Eigen::Vector3d>& a, const std::vector<Eigen::Vector3d>& b) {
    Rigid out;
    if (a.size() < 3 || a.size() != b.size()) return out;
    Eigen::Vector3d ca = Eigen::Vector3d::Zero(), cb = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < a.size(); ++i) { ca += a[i]; cb += b[i]; }
    ca /= static_cast<double>(a.size());
    cb /= static_cast<double>(b.size());

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (std::size_t i = 0; i < a.size(); ++i) H += (a[i] - ca) * (b[i] - cb).transpose();

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    if ((svd.matrixV() * svd.matrixU().transpose()).determinant() < 0.0) D(2, 2) = -1.0;
    out.R = svd.matrixV() * D * svd.matrixU().transpose();
    out.t = cb - out.R * ca;
    return out;
}

// 가장 가까운 시각의 인덱스
int nearestIdx(const std::vector<Pose>& v, double t) {
    if (v.empty()) return -1;
    int best = 0;
    double bd = std::abs(v[0].t - t);
    for (std::size_t i = 1; i < v.size(); ++i) {
        const double d = std::abs(v[i].t - t);
        if (d < bd) { bd = d; best = static_cast<int>(i); }
    }
    return best;
}

// 정답 궤적의 주성분 2 개로 투영면을 정한다.
// TUM 은 손에 든 카메라라 3D 로 돌아다니고 KITTI 는 도로 평면 위를 간다.
// 축을 고정하면 한쪽이 선 하나로 뭉개진다.
struct Plane2 {
    Eigen::Vector3d e1{Eigen::Vector3d::UnitX()}, e2{Eigen::Vector3d::UnitZ()};
    Eigen::Vector3d c{Eigen::Vector3d::Zero()};
};

Plane2 principalPlane(const std::vector<Pose>& gt) {
    Plane2 out;
    if (gt.size() < 3) return out;
    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    for (const auto& p : gt) c += p.p;
    c /= static_cast<double>(gt.size());
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
    for (const auto& p : gt) { const Eigen::Vector3d d = p.p - c; C += d * d.transpose(); }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);
    out.c = c;
    out.e1 = es.eigenvectors().col(2);   // 최대 분산
    out.e2 = es.eigenvectors().col(1);
    return out;
}

cv::Point2f project(const Plane2& pl, const Eigen::Vector3d& p) {
    const Eigen::Vector3d d = p - pl.c;
    return {static_cast<float>(d.dot(pl.e1)), static_cast<float>(d.dot(pl.e2))};
}

// --- 그리기 유틸 ------------------------------------------------------------

void text(cv::Mat& c, const std::string& s, cv::Point at, double sc,
          const cv::Scalar& col, int th = 1) {
    cv::putText(c, s, at, cv::FONT_HERSHEY_SIMPLEX, sc, col, th, cv::LINE_AA);
}

int textW(const std::string& s, double sc, int th = 1) {
    int base = 0;
    return cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, sc, th, &base).width;
}

void panel(cv::Mat& c, cv::Rect r, const cv::Scalar& col, int radius = 6) {
    cv::Mat roi = c(r & cv::Rect(0, 0, c.cols, c.rows));
    roi.setTo(col);
    (void)radius;
}

std::string fmt(double v, int nd, const char* unit = "") {
    if (!std::isfinite(v)) return "n/a";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(nd) << v << unit;
    return ss.str();
}

// 궤적을 주어진 사각형 안에 그린다. 두 패널이 **같은 스케일**을 쓰도록
// 바깥에서 계산한 범위를 받는다 - 각자 자동맞춤하면 더 나쁜 쪽이 더 좋아 보인다.
struct View2 {
    cv::Point2f mn{0, 0}, mx{0, 0};
    bool valid{false};
};

void growView(View2& v, const std::vector<cv::Point2f>& pts) {
    for (const auto& p : pts) {
        if (!v.valid) { v.mn = v.mx = p; v.valid = true; continue; }
        v.mn.x = std::min(v.mn.x, p.x); v.mn.y = std::min(v.mn.y, p.y);
        v.mx.x = std::max(v.mx.x, p.x); v.mx.y = std::max(v.mx.y, p.y);
    }
}

void drawTraj(cv::Mat& c, cv::Rect r, const View2& v,
              const std::vector<cv::Point2f>& gt,
              const std::vector<cv::Point2f>& est,
              const cv::Scalar& est_col, int upto) {
    panel(c, r, kPanel);
    cv::rectangle(c, r, kInkFaint, 1, cv::LINE_AA);
    if (!v.valid) return;

    const float sx = static_cast<float>(r.width - 24) /
                     std::max(1e-6f, v.mx.x - v.mn.x);
    const float sy = static_cast<float>(r.height - 24) /
                     std::max(1e-6f, v.mx.y - v.mn.y);
    const float s = std::min(sx, sy);
    const float ox = r.x + r.width * 0.5f - (v.mn.x + v.mx.x) * 0.5f * s;
    const float oy = r.y + r.height * 0.5f + (v.mn.y + v.mx.y) * 0.5f * s;
    auto map = [&](const cv::Point2f& p) {
        return cv::Point2f(ox + p.x * s, oy - p.y * s);   // y 뒤집기
    };

    // 축척 막대. 숫자 없이 그림만 있으면 크기를 알 수 없다.
    {
        const double span = (v.mx.x - v.mn.x);
        double unit = std::pow(10.0, std::floor(std::log10(std::max(1e-6, span * 0.3))));
        if (span / unit > 6.0) unit *= 2.0;
        const int px = static_cast<int>(unit * s);
        if (px > 12 && px < r.width - 40) {
            const cv::Point a(r.x + 12, r.y + r.height - 12);
            cv::line(c, a, {a.x + px, a.y}, kInkDim, 1, cv::LINE_AA);
            cv::line(c, {a.x, a.y - 3}, {a.x, a.y + 3}, kInkDim, 1, cv::LINE_AA);
            cv::line(c, {a.x + px, a.y - 3}, {a.x + px, a.y + 3}, kInkDim, 1, cv::LINE_AA);
            text(c, fmt(unit, unit < 1.0 ? 2 : 0, " m"), {a.x, a.y - 7}, 0.36, kInkDim);
        }
    }

    for (std::size_t i = 1; i < gt.size(); ++i) {
        cv::line(c, map(gt[i - 1]), map(gt[i]), kGt, 1, cv::LINE_AA);
    }
    const int n = std::min<int>(upto, static_cast<int>(est.size()));
    for (int i = 1; i < n; ++i) {
        cv::line(c, map(est[i - 1]), map(est[i]), est_col, 2, cv::LINE_AA);
    }
    if (n > 0) cv::circle(c, map(est[n - 1]), 4, est_col, -1, cv::LINE_AA);
}

// --- 매니페스트 -------------------------------------------------------------

bool loadManifest(const fs::path& p, std::vector<Seq>& seqs) {
    std::ifstream f(p);
    if (!f) return false;
    std::map<std::string, std::size_t> idx;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto c = split(line, '\t');
        if (c.empty()) continue;
        if (c[0] == "SEQ" && c.size() >= 6) {
            Seq s;
            s.name = c[1]; s.dataset = c[2]; s.dir = c[3]; s.gt_file = c[4];
            s.identity_ate_cm = toD(c[5]);
            idx[s.name] = seqs.size();
            seqs.push_back(std::move(s));
        } else if (c[0] == "RUN" && c.size() >= 11) {
            auto it = idx.find(c[1]);
            if (it == idx.end()) continue;
            Run r;
            r.system = c[2]; r.label = c[3]; r.kind = c[4];
            r.ate_cm = toD(c[5]); r.rpe_mm = toD(c[6]); r.ms = toD(c[7]);
            r.frames = std::atoi(c[8].c_str());
            r.status = c[9]; r.traj_file = c[10];
            seqs[it->second].runs[r.system] = std::move(r);
        }
    }
    return !seqs.empty();
}

void loadSeq(Seq& s) {
    if (s.loaded) return;
    s.gt = readTum(s.gt_file);
    s.rgb = readIndex(s.dir, "rgb.txt");
    const Plane2 pl = principalPlane(s.gt);

    s.gt_xy.clear();
    s.gt_xy.reserve(s.gt.size());
    for (const auto& g : s.gt) s.gt_xy.push_back(project(pl, g.p));

    for (auto& [key, r] : s.runs) {
        r.traj = readTum(r.traj_file);
        // 표시용 강체 정렬: 추정 시각에 대응하는 정답 위치를 모아 Kabsch.
        std::vector<Eigen::Vector3d> A, B;
        for (const auto& e : r.traj) {
            const int j = nearestIdx(s.gt, e.t);
            if (j < 0 || std::abs(s.gt[static_cast<std::size_t>(j)].t - e.t) > 0.25) continue;
            A.push_back(e.p);
            B.push_back(s.gt[static_cast<std::size_t>(j)].p);
        }
        const Rigid T = kabsch(A, B);
        r.xy.clear();
        r.xy.reserve(r.traj.size());
        for (const auto& e : r.traj) r.xy.push_back(project(pl, T.R * e.p + T.t));
    }
    s.loaded = true;
}

}  // namespace

int main(int argc, char** argv) {
    fs::path manifest = "results/bench/viewer.tsv";
    int start_seq = 0;
    bool autoplay = true;
    // 창을 띄우지 않고 한 장만 렌더링해 저장한다. 문서용 그림을 만들 때와,
    // 화면 없는 환경에서 레이아웃이 깨지지 않았는지 확인할 때 쓴다.
    std::string shot_path;
    int shot_frame = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--manifest") manifest = argv[i + 1];
        else if (k == "--seq") start_seq = std::atoi(argv[i + 1]);
        else if (k == "--autoplay") autoplay = std::atoi(argv[i + 1]) != 0;
        else if (k == "--screenshot") shot_path = argv[i + 1];
        else if (k == "--frame") shot_frame = std::atoi(argv[i + 1]);
    }

    std::vector<Seq> seqs;
    if (!loadManifest(manifest, seqs)) {
        std::cerr <<
            "매니페스트를 읽지 못했다: " << manifest << "\n"
            "  python tools/bench_run.py     (두 시스템 실행 + 채점)\n"
            "  python tools/bench_export.py  (-> results/bench/viewer.tsv)\n"
            "지표를 여기서 다시 계산하지 않으므로, 없으면 실행하지 않는다.\n";
        return 1;
    }
    std::cout << "시퀀스 " << seqs.size() << " 개\n";

    const int W = 1600, H = 920;
    const std::string win = "WorldVision-SLAM  -  Benchmark Viewer";
    const bool headless = !shot_path.empty();
    if (!headless) cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

    int si = std::clamp(start_seq, 0, static_cast<int>(seqs.size()) - 1);
    int frame = headless ? shot_frame : 0;
    bool playing = autoplay;
    int shot = 0;

    cv::Mat canvas(H, W, CV_8UC3);

    while (true) {
        Seq& s = seqs[static_cast<std::size_t>(si)];
        loadSeq(s);

        const Run* L = s.runs.count("baseline") ? &s.runs.at("baseline") : nullptr;
        const Run* R = s.runs.count("wme") ? &s.runs.at("wme") : nullptr;

        const int nframes = std::max<int>(1, static_cast<int>(s.rgb.size()));
        frame = std::clamp(frame, 0, nframes - 1);

        // 두 패널이 같은 스케일을 쓰도록 범위를 함께 잡는다.
        View2 view;
        growView(view, s.gt_xy);
        if (L) growView(view, L->xy);
        if (R) growView(view, R->xy);

        canvas.setTo(kBg);

        // ---- 머리말 ----
        panel(canvas, {0, 0, W, 62}, kPanelHi);
        text(canvas, "WorldVision-SLAM", {20, 40}, 0.82, kInk, 2);
        text(canvas, "Benchmark Viewer", {320, 40}, 0.62, kInkDim, 1);
        {
            std::ostringstream ss;
            ss << "[" << (si + 1) << "/" << seqs.size() << "]  "
               << s.name << "   (" << s.dataset << ")";
            const std::string t = ss.str();
            text(canvas, t, {W - 30 - textW(t, 0.6, 1), 40}, 0.6, kInk, 1);
        }

        // ---- 두 패널 ----
        const int py = 72, ph = H - 72 - 104, pw = (W - 36) / 2;
        struct Side { const Run* run; cv::Scalar col; int x; const char* tag; };
        const Side sides[2] = {
            {L, kLeft,  12,           "LEFT   classical descriptor pipeline"},
            {R, kRight, 24 + pw,      "RIGHT  WME  (descriptor-free)"},
        };

        for (int k = 0; k < 2; ++k) {
            const Side& sd = sides[k];
            const cv::Rect box(sd.x, py, pw, ph);
            panel(canvas, box, kPanel);
            cv::rectangle(canvas, box, kInkFaint, 1, cv::LINE_AA);
            cv::line(canvas, {box.x, box.y}, {box.x + box.width, box.y}, sd.col, 3, cv::LINE_AA);

            text(canvas, sd.tag, {box.x + 16, box.y + 30}, 0.46, kInkDim);
            text(canvas, sd.run ? sd.run->label : "(no run)",
                 {box.x + 16, box.y + 58}, 0.66, sd.col, 2);

            // 영상
            const cv::Rect vid(box.x + 16, box.y + 74, box.width - 32, 300);
            panel(canvas, vid, kBg);
            if (!s.rgb.empty()) {
                cv::Mat img = cv::imread(s.rgb[static_cast<std::size_t>(frame)].second,
                                         cv::IMREAD_COLOR);
                if (!img.empty()) {
                    const double sc = std::min(static_cast<double>(vid.width) / img.cols,
                                               static_cast<double>(vid.height) / img.rows);
                    cv::Mat small;
                    cv::resize(img, small, {}, sc, sc, cv::INTER_AREA);
                    const int ox = vid.x + (vid.width - small.cols) / 2;
                    const int oy = vid.y + (vid.height - small.rows) / 2;
                    small.copyTo(canvas(cv::Rect(ox, oy, small.cols, small.rows)));
                    cv::rectangle(canvas, cv::Rect(ox, oy, small.cols, small.rows),
                                  kInkFaint, 1, cv::LINE_AA);
                }
            }

            // 궤적
            const cv::Rect tr(box.x + 16, vid.y + vid.height + 12,
                              box.width - 32, ph - 74 - 300 - 12 - 108);
            const double frac = static_cast<double>(frame + 1) / nframes;
            const int upto = sd.run
                ? std::max(1, static_cast<int>(frac * static_cast<double>(sd.run->xy.size())))
                : 0;
            drawTraj(canvas, tr, view, s.gt_xy, sd.run ? sd.run->xy : std::vector<cv::Point2f>{},
                     sd.col, upto);
            text(canvas, "ground truth", {tr.x + 12, tr.y + 20}, 0.38, kGt);
            text(canvas, "estimate (rigidly aligned for display)",
                 {tr.x + 12, tr.y + 38}, 0.38, sd.col);

            // 지표
            const int my = tr.y + tr.height + 30;
            const bool has = sd.run != nullptr;
            const double ate = has ? sd.run->ate_cm : std::nan("");
            const double other = (k == 0 ? (R ? R->ate_cm : std::nan(""))
                                         : (L ? L->ate_cm : std::nan("")));
            const bool better = std::isfinite(ate) && std::isfinite(other) && ate < other;

            text(canvas, "ATE RMSE", {box.x + 16, my}, 0.44, kInkDim);
            text(canvas, fmt(ate, 2, " cm"), {box.x + 16, my + 34}, 0.95,
                 better ? kWin : kInk, 2);
            if (better) {
                const std::string b = "BETTER";
                const int bx = box.x + 20 + textW(fmt(ate, 2, " cm"), 0.95, 2);
                panel(canvas, {bx, my + 12, textW(b, 0.4, 1) + 16, 24}, kWin);
                text(canvas, b, {bx + 8, my + 29}, 0.4, kBg, 1);
            }

            text(canvas, "RPE trans (median)", {box.x + 280, my}, 0.44, kInkDim);
            text(canvas, has ? fmt(sd.run->rpe_mm, 2, " mm") : "n/a",
                 {box.x + 280, my + 30}, 0.6, kInk, 1);

            text(canvas, "ms / frame", {box.x + 540, my}, 0.44, kInkDim);
            text(canvas, has ? fmt(sd.run->ms, 1, "") : "n/a",
                 {box.x + 540, my + 30}, 0.6, kInk, 1);

            if (has && sd.run->status != "ok") {
                const std::string w = "STATUS: " + sd.run->status + "  (ATE is meaningless)";
                text(canvas, w, {box.x + 16, my + 58}, 0.45, kLose, 1);
            }
        }

        // ---- 바닥 ----
        const int fy = H - 100;
        panel(canvas, {0, fy, W, 100}, kPanelHi);
        {
            // 진행 막대
            const cv::Rect bar(20, fy + 14, W - 40, 8);
            panel(canvas, bar, kBg);
            const int done = static_cast<int>(bar.width *
                             (static_cast<double>(frame + 1) / nframes));
            panel(canvas, {bar.x, bar.y, std::max(2, done), bar.height}, kInkDim);

            std::ostringstream ss;
            ss << "frame " << (frame + 1) << " / " << nframes;
            text(canvas, ss.str(), {20, fy + 46}, 0.48, kInk);

            const std::string floor_s =
                "do-nothing floor  " + fmt(s.identity_ate_cm, 2, " cm");
            text(canvas, floor_s, {240, fy + 46}, 0.48, kInkDim);

            text(canvas, playing ? "PLAYING" : "PAUSED",
                 {W - 30 - textW(playing ? "PLAYING" : "PAUSED", 0.48, 1), fy + 46},
                 0.48, playing ? kWin : kInkDim);

            text(canvas,
                 "SPACE play/pause   <- -> step   N/P sequence   R restart   "
                 "S screenshot   Q quit",
                 {20, fy + 76}, 0.44, kInkDim);
            const std::string src = "metrics from results/bench/viewer.tsv "
                                    "(computed by tools/bench_run.py, not here)";
            text(canvas, src, {W - 30 - textW(src, 0.38, 1), fy + 76}, 0.38, kInkFaint);
        }

        if (headless) {
            if (!cv::imwrite(shot_path, canvas)) {
                std::cerr << "스크린샷 저장 실패: " << shot_path << "\n";
                return 1;
            }
            std::cout << "저장: " << shot_path << "  (" << s.name << ", frame "
                      << (frame + 1) << "/" << nframes << ")\n";
            return 0;
        }

        cv::imshow(win, canvas);
        const int key = cv::waitKey(playing ? 30 : 0);

        if (key == 'q' || key == 'Q' || key == 27) break;
        else if (key == ' ') playing = !playing;
        else if (key == 'r' || key == 'R') frame = 0;
        else if (key == 'n' || key == 'N') {
            si = (si + 1) % static_cast<int>(seqs.size()); frame = 0;
        } else if (key == 'p' || key == 'P') {
            si = (si + static_cast<int>(seqs.size()) - 1) % static_cast<int>(seqs.size());
            frame = 0;
        } else if (key == 's' || key == 'S') {
            const std::string f = "viewer_" + s.name + "_" + std::to_string(shot++) + ".png";
            if (cv::imwrite(f, canvas)) std::cout << "저장: " << f << "\n";
            else std::cerr << "스크린샷 저장 실패: " << f << "\n";
        } else if (key == 81 || key == 2424832) frame = std::max(0, frame - 1);
        else if (key == 83 || key == 2555904) frame = std::min(nframes - 1, frame + 1);
        else if (playing) {
            if (++frame >= nframes) { frame = 0; }
        }

        // 창이 닫히면 종료 (X 버튼)
        if (cv::getWindowProperty(win, cv::WND_PROP_VISIBLE) < 1.0) break;
    }

    cv::destroyAllWindows();
    return 0;
}
