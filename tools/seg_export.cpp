// WorldVision-SLAM — 의미 분할 내보내기
//
// 프레임마다 픽셀 클래스를 미리 계산해 파일로 남긴다. 뷰어는 그것을 읽기만
// 한다 - scene_export 가 검출 상자를 미리 내보내는 것과 같은 구조다.
//
// **왜 뷰어 안에서 돌리지 않는가.** SegFormer-B0 512x512 는 CPU 에서 프레임당
// 수십 ms 다. 뷰어는 이미 라벨 62 ms + 그리기 30 ms 를 쓰고 있고, 거기에
// 추론을 얹으면 화면이 눈에 띄게 끊긴다. 한 번 계산해서 재생 때 읽으면
// 재생은 공짜이고, 같은 답을 매번 다시 구할 이유도 없다.
//
// **왜 기하가 아니라 외형인가.** 지금까지 뷰어는 복셀의 국소 구조(평면성·
// 선형성·산포)로 클래스를 정했다. 그 방법으로는 갈리지 않는 쌍이 있다:
// 옆에서 본 벽 조각과 나무줄기 줄은 국소 구조가 같고(실측: vert p75 가
// 겹치고 scatter 는 소수점 셋째 자리까지 동일), 아스팔트와 잔디밭은 둘 다
// 평평한 수평면이다. 문턱을 아무리 옮겨도 한 데이터셋의 오류가 다른 쪽으로
// 옮겨갈 뿐이었다.
//
// 카메라 화소는 그 둘을 안다. 아스팔트는 회색이고 잔디는 밝기·질감이 다르며,
// 벽면은 잎과 다르게 생겼다. Cityscapes 로 학습한 분할망은 정확히 그
// 구분을 배운 것이고, KITTI 는 같은 종류의 도로 장면이다.
//
// 모델: segformer-b0-finetuned-cityscapes-1024-1024 (ONNX).
// models/*.onnx 는 .gitignore 에 있으므로 저장소에 들어가지 않는다 -
// yolo11n.onnx 와 같은 취급이다.
//
// 출력: <out>/<프레임번호 6자리>.png, 8 비트 단일 채널, 원본의 1/4 해상도.
//       화소값이 곧 Cityscapes 클래스 번호이고 255 는 "판정 없음" 이다.

#include "dataset_calib.hpp"

#include <onnxruntime_cxx_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Cityscapes 학습 클래스 19 종. 인덱스가 곧 모델 출력 채널이다.
//
//  0 road        1 sidewalk   2 building   3 wall       4 fence
//  5 pole        6 tlight     7 tsign      8 vegetation 9 terrain
// 10 sky        11 person    12 rider     13 car       14 truck
// 15 bus        16 train     17 motorcycle 18 bicycle
constexpr int kNumClass = 19;

const char* kClassName[kNumClass] = {
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle"};

#ifdef _WIN32
std::wstring widen(const std::string& s) {
    return std::wstring(s.begin(), s.end());
}
#endif

// 전처리는 preprocessor_config.json 그대로다. 여기서 임의로 바꾸면 학습
// 분포와 어긋나 조용히 나빠진다 - 크기 512, 1/255, ImageNet 평균·표준편차.
constexpr int   kInW = 512, kInH = 512;
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3]  = {0.229f, 0.224f, 0.225f};

std::vector<std::string> listFrames(const fs::path& dir) {
    std::vector<std::string> out;
    if (!fs::exists(dir)) return out;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        const auto ext = e.path().extension().string();
        if (ext == ".png" || ext == ".jpg") out.push_back(e.path().string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string seq_dir, out_dir;
    std::string model = "models/segformer_b0_cityscapes.onnx";
    int stride = 1, max_frames = 0;

    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--seq")             seq_dir = argv[i + 1];
        else if (k == "--out")        out_dir = argv[i + 1];
        else if (k == "--model")      model = argv[i + 1];
        else if (k == "--stride")     stride = std::max(1, std::atoi(argv[i + 1]));
        else if (k == "--max-frames") max_frames = std::atoi(argv[i + 1]);
    }
    if (seq_dir.empty() || out_dir.empty()) {
        std::cerr <<
            "사용법: wme_seg_export --seq <시퀀스디렉터리> --out <출력디렉터리>\n"
            "              [--model models/segformer_b0_cityscapes.onnx]\n"
            "              [--stride 1] [--max-frames 0]\n";
        return 2;
    }
    if (!fs::exists(model)) {
        std::cerr << "모델이 없다: " << model << "\n";
        return 1;
    }

    const auto frames = listFrames(fs::path(seq_dir) / "rgb");
    if (frames.empty()) {
        std::cerr << "프레임이 없다: " << (fs::path(seq_dir) / "rgb").string() << "\n";
        return 1;
    }
    fs::create_directories(out_dir);

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "wme_seg"};
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(static_cast<int>(
        std::max(1u, std::thread::hardware_concurrency() / 2)));
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
    Ort::Session session(env, widen(model).c_str(), opts);
#else
    Ort::Session session(env, model.c_str(), opts);
#endif
    Ort::AllocatorWithDefaultOptions alloc;
    const std::string in_name  = session.GetInputNameAllocated(0, alloc).get();
    const std::string out_name = session.GetOutputNameAllocated(0, alloc).get();
    const char* in_names[]  = {in_name.c_str()};
    const char* out_names[] = {out_name.c_str()};

    std::cout << "모델: " << model << "  입력 " << in_name
              << "  출력 " << out_name << "\n"
              << "프레임 " << frames.size() << " (stride " << stride << ")\n";

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<float> blob(static_cast<std::size_t>(3) * kInW * kInH);
    const std::array<std::int64_t, 4> in_shape{1, 3, kInH, kInW};

    // 클래스별 화소 수. 끝에 찍어서 결과가 말이 되는지 바로 본다 -
    // 도로 장면인데 road 가 0 이면 전처리가 어긋난 것이다.
    std::vector<long long> tally(kNumClass, 0);
    int done = 0;

    for (std::size_t fi = 0; fi < frames.size(); fi += static_cast<std::size_t>(stride)) {
        if (max_frames > 0 && done >= max_frames) break;
        const cv::Mat bgr = cv::imread(frames[fi], cv::IMREAD_COLOR);
        if (bgr.empty()) continue;

        cv::Mat rz;
        cv::resize(bgr, rz, cv::Size(kInW, kInH), 0, 0, cv::INTER_LINEAR);
        // BGR -> RGB, 1/255, 정규화, NCHW
        for (int y = 0; y < kInH; ++y) {
            const auto* row = rz.ptr<cv::Vec3b>(y);
            for (int x = 0; x < kInW; ++x) {
                for (int c = 0; c < 3; ++c) {
                    const float v = static_cast<float>(row[x][2 - c]) / 255.0f;
                    blob[static_cast<std::size_t>(c) * kInH * kInW
                         + static_cast<std::size_t>(y) * kInW + x] =
                        (v - kMean[c]) / kStd[c];
                }
            }
        }

        Ort::Value in = Ort::Value::CreateTensor<float>(
            mem, blob.data(), blob.size(), in_shape.data(), in_shape.size());
        auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);

        const auto info = outs[0].GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape();          // [1, 19, h, w]
        if (shape.size() != 4 || shape[1] != kNumClass) {
            std::cerr << "예상 못 한 출력 모양: [";
            for (auto d : shape) std::cerr << d << ' ';
            std::cerr << "]\n";
            return 1;
        }
        const int oh = static_cast<int>(shape[2]), ow = static_cast<int>(shape[3]);
        const float* logit = outs[0].GetTensorData<float>();

        // 채널 방향 argmax. 로짓이므로 softmax 는 필요 없다.
        cv::Mat lab(oh, ow, CV_8UC1);
        const std::size_t plane = static_cast<std::size_t>(oh) * ow;
        for (int y = 0; y < oh; ++y) {
            auto* lr = lab.ptr<std::uint8_t>(y);
            for (int x = 0; x < ow; ++x) {
                const std::size_t at = static_cast<std::size_t>(y) * ow + x;
                int best = 0;
                float bv = logit[at];
                for (int c = 1; c < kNumClass; ++c) {
                    const float v = logit[c * plane + at];
                    if (v > bv) { bv = v; best = c; }
                }
                lr[x] = static_cast<std::uint8_t>(best);
            }
        }

        // 원본의 1/4 로 되돌린다. 뷰어가 화소 -> 라벨을 찾을 때 배율 하나로
        // 끝나도록 원본 비율을 그대로 따른다. 최근접이어야 한다 - 라벨을
        // 선형 보간하면 도로와 인도 사이에 없는 클래스가 생긴다.
        cv::Mat outlab;
        cv::resize(lab, outlab, cv::Size(bgr.cols / 4, bgr.rows / 4), 0, 0,
                   cv::INTER_NEAREST);
        for (int y = 0; y < outlab.rows; ++y) {
            const auto* r = outlab.ptr<std::uint8_t>(y);
            for (int x = 0; x < outlab.cols; ++x) {
                if (r[x] < kNumClass) ++tally[r[x]];
            }
        }

        const fs::path dst = fs::path(out_dir) /
            (fs::path(frames[fi]).stem().string() + ".png");
        cv::imwrite(dst.string(), outlab);
        ++done;
        if (done % 25 == 0) {
            std::cout << "  " << done << " 프레임\r" << std::flush;
        }
    }

    long long total = 0;
    for (long long t : tally) total += t;
    std::cout << "\n완료: " << done << " 프레임 -> " << out_dir << "\n";
    if (total > 0) {
        std::cout << "클래스 분포:\n";
        for (int c = 0; c < kNumClass; ++c) {
            if (tally[c] == 0) continue;
            std::cout << "  " << kClassName[c] << "  "
                      << (100.0 * static_cast<double>(tally[c])
                          / static_cast<double>(total)) << " %\n";
        }
    }
    return 0;
}
