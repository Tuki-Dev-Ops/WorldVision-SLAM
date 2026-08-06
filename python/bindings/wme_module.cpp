// C++ 엔진의 pybind11 바인딩.
//
// 목적은 두 가지다.
//   1. 차등 테스트 - Python 참조 구현과 같은 입력에 같은 답을 내는지 확인
//   2. 평가 하네스 - 데이터셋 반복과 지표 계산은 Python 이 훨씬 빠르게 쓰인다
//
// 바인딩은 얇게 유지한다. 로직은 전부 C++ 쪽에 있어야 하고 여기는 변환만 한다.

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "wme/confidence/ConfidenceEngine.hpp"
#include "wme/core/Assignment.hpp"
#include "wme/core/SE3.hpp"
#include "wme/fusion/PoseFusion.hpp"
#include "wme/geometry/PlaneExtractor.hpp"
#include "wme/geometry/StructuralAligner.hpp"
#include "wme/localization/DirectAligner.hpp"
#include "wme/perception/Detection.hpp"
#include "wme/perception/EnvironmentAnalyzer.hpp"
#include "wme/perception/ImageQualityEngine.hpp"
#include "wme/perception/StereoDepth.hpp"
#include "wme/token/ConstellationIndex.hpp"
#include "wme/token/TokenStore.hpp"

#include <opencv2/imgproc.hpp>

#include <cstring>
#include <stdexcept>

namespace py = pybind11;
using namespace wme;

namespace {

// numpy (H, W) uint8 / (H, W, 3) uint8 / (H, W) float32 -> cv::Mat (복사 없음)
//
// cv::Mat 은 행 안이 빈틈없이 이어져 있다고 가정한다. 행 사이 패딩(step)만
// 표현할 수 있고 원소 사이 간격은 표현할 수 없다. 그래서 numpy 의 마지막
// 축 stride 가 itemsize 가 아니면 이 변환은 *다른 픽셀* 을 읽는다.
//
// OpenCV 는 step < 최소행길이 인 경우만 assert 로 잡는다(F-order 는 여기서
// 걸린다). img[:, ::2] 처럼 열 방향 슬라이스는 step 이 오히려 커서 그 검사를
// 통과하고, 예외 없이 조용히 틀린 답을 낸다 - 실측으로 같은 픽셀 집합에
// 대해 품질점수가 0.974 대신 0.898 이 나왔다. 마지막 축을 여기서 막는다.
cv::Mat toMat(const py::array& arr) {
    py::buffer_info info = arr.request();
    if (info.ndim < 2 || info.ndim > 3) {
        throw std::invalid_argument("2D 또는 3D 배열이어야 함");
    }
    const int rows = static_cast<int>(info.shape[0]);
    const int cols = static_cast<int>(info.shape[1]);
    const int ch = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;

    int depth;
    if (info.format == py::format_descriptor<std::uint8_t>::format())      depth = CV_8U;
    else if (info.format == py::format_descriptor<float>::format())        depth = CV_32F;
    else if (info.format == py::format_descriptor<double>::format())       depth = CV_64F;
    else throw std::invalid_argument("uint8 / float32 / float64 만 지원");

    // 마지막 축부터 안쪽으로 연속인지 확인한다. 행 사이 패딩(strides[0])만 허용.
    if (info.strides[info.ndim - 1] != info.itemsize) {
        throw std::invalid_argument(
            "마지막 축이 연속이어야 함 - np.ascontiguousarray 로 감싸라");
    }
    if (info.ndim == 3 && info.strides[1] != info.itemsize * ch) {
        throw std::invalid_argument("채널 축이 연속이어야 함");
    }
    if (info.strides[0] < static_cast<py::ssize_t>(info.itemsize) * cols * ch) {
        throw std::invalid_argument("행 stride 가 행 길이보다 작음");
    }

    return cv::Mat(rows, cols, CV_MAKETYPE(depth, ch), info.ptr, info.strides[0]);
}

py::array_t<float> fromMat32F(const cv::Mat& m) {
    if (m.empty()) return py::array_t<float>();
    cv::Mat f;
    m.convertTo(f, CV_32F);
    py::array_t<float> out({f.rows, f.cols});
    std::memcpy(out.mutable_data(), f.data, static_cast<std::size_t>(f.total()) * sizeof(float));
    return out;
}

// Result<T> -> 값 또는 예외. Python 쪽에서는 예외가 자연스럽다.
template <typename T>
T unwrap(Result<T> r) {
    if (!r) throw std::runtime_error(r.error().message());
    return std::move(r).value();
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "WME C++ engine bindings";
    m.attr("__version__") = "0.1.0";

    // --- SE3 --------------------------------------------------------------
    py::class_<SO3>(m, "SO3")
        .def(py::init<>())
        .def(py::init([](const Mat3& R) { return SO3(R); }))
        .def_static("exp", &SO3::exp)
        .def("log", &SO3::log)
        .def("matrix", &SO3::matrix)
        .def("inverse", &SO3::inverse)
        .def_static("left_jacobian", &SO3::leftJacobian)
        .def_static("left_jacobian_inv", &SO3::leftJacobianInverse);

    py::class_<SE3>(m, "SE3")
        .def(py::init<>())
        .def(py::init([](const Mat3& R, const Vec3& t) { return SE3(R, t); }))
        .def_static("exp", &SE3::exp)
        .def_static("identity", &SE3::identity)
        .def("log", &SE3::log)
        .def("matrix", &SE3::matrix)
        .def("inverse", &SE3::inverse)
        .def("adjoint", &SE3::adjoint)
        .def_property_readonly("t", [](const SE3& T) { return T.translation(); })
        .def_property_readonly("R", [](const SE3& T) { return T.rotation().matrix(); })
        .def("distance_to", [](const SE3& a, const SE3& b) {
            const Vec2 d = a.distanceTo(b);
            return py::make_tuple(d.x(), d.y());
        })
        .def("__matmul__", [](const SE3& a, const SE3& b) { return a * b; })
        .def("__matmul__", [](const SE3& a, const Vec3& p) { return a * p; });

    m.def("kabsch", [](const std::vector<Vec3>& src, const std::vector<Vec3>& dst) {
        return unwrap(kabsch(src, dst));
    }, py::arg("src"), py::arg("dst"));

    // --- 할당 -------------------------------------------------------------
    m.attr("INFEASIBLE") = kInfeasible;
    m.def("solve_assignment", [](const py::array_t<double, py::array::c_style>& cost) {
        const auto info = cost.request();
        if (info.ndim != 2) throw std::invalid_argument("2D 비용행렬이어야 함");
        const auto rows = static_cast<std::size_t>(info.shape[0]);
        const auto cols = static_cast<std::size_t>(info.shape[1]);

        const std::vector<double> flat(cost.data(), cost.data() + rows * cols);
        const AssignmentResult r = solveAssignment(flat, rows, cols);
        return py::make_tuple(r.row_to_col, r.col_to_row, r.total_cost);
    }, py::arg("cost"));

    // --- 성좌 -------------------------------------------------------------
    py::class_<ConstellationNode>(m, "ConstellationNode")
        .def(py::init([](std::uint64_t id, int cls, const Vec3& p, double sigma) {
            ConstellationNode n;
            n.id = TokenId(id);
            n.class_id = cls;
            n.position = p;
            n.sigma = sigma;
            return n;
        }), py::arg("token_id"), py::arg("class_id"), py::arg("position"),
            py::arg("sigma") = 0.05)
        .def_property_readonly("token_id", [](const ConstellationNode& n) { return n.id.value; })
        .def_readwrite("class_id", &ConstellationNode::class_id)
        .def_readwrite("position", &ConstellationNode::position)
        .def_readwrite("sigma", &ConstellationNode::sigma);

    py::class_<ConstellationMatch>(m, "ConstellationMatch")
        .def_readonly("place_id", &ConstellationMatch::place_id)
        .def_readonly("transform", &ConstellationMatch::transform)
        .def_readonly("rms_error", &ConstellationMatch::rms_error)
        .def_readonly("score", &ConstellationMatch::score)
        // 수용 판정에 쓰이는 진단은 전부 노출한다. 하나라도 빠지면 차등 테스트가
        // 그 항을 비교할 수 없고, 비교하지 못하는 항이 곧 다음 불일치가 된다.
        .def_readonly("n_query_nodes", &ConstellationMatch::n_query_nodes)
        .def_readonly("n_place_nodes", &ConstellationMatch::n_place_nodes)
        .def_readonly("n_inliers", &ConstellationMatch::n_inliers)
        .def_readonly("explained", &ConstellationMatch::explained)
        .def_readonly("chi2_dof", &ConstellationMatch::chi2_dof)
        .def_readonly("agree_count", &ConstellationMatch::agree_count)
        .def_readonly("support", &ConstellationMatch::support)
        .def_readonly("rival_mass", &ConstellationMatch::rival_mass)
        .def_readonly("pose_margin", &ConstellationMatch::pose_margin)
        .def_readonly("confidence", &ConstellationMatch::confidence)
        .def_property_readonly("correspondences", [](const ConstellationMatch& m2) {
            std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
            out.reserve(m2.correspondences.size());
            for (const auto& [a, b] : m2.correspondences) out.emplace_back(a.value, b.value);
            return out;
        });

    py::class_<ConstellationConfig>(m, "ConstellationConfig")
        .def(py::init<>())
        .def_readwrite("pose_dominance", &ConstellationConfig::pose_dominance)
        .def_readwrite("chi2_confidence_scale", &ConstellationConfig::chi2_confidence_scale)
        .def_readwrite("min_confidence", &ConstellationConfig::min_confidence)
        .def_readwrite("min_nodes", &ConstellationConfig::min_nodes)
        .def_readwrite("distance_tolerance", &ConstellationConfig::distance_tolerance)
        .def_readwrite("relative_tolerance", &ConstellationConfig::relative_tolerance)
        .def_readwrite("sigma_gate", &ConstellationConfig::sigma_gate)
        .def_readwrite("max_rms_error", &ConstellationConfig::max_rms_error)
        .def_readwrite("use_chirality", &ConstellationConfig::use_chirality);

    py::class_<ConstellationIndex>(m, "ConstellationIndex")
        .def(py::init<ConstellationConfig>(), py::arg("config") = ConstellationConfig{})
        .def("insert", [](ConstellationIndex& self, std::uint64_t kf, double stamp,
                          const SE3& anchor, std::vector<ConstellationNode> nodes,
                          std::optional<Vec3> gravity) {
            return self.insert(KeyframeId(kf), Timestamp::fromSeconds(stamp), anchor,
                               std::move(nodes), gravity);
        }, py::arg("keyframe"), py::arg("stamp"), py::arg("anchor"), py::arg("nodes"),
           py::arg("gravity") = py::none())
        .def("query", [](const ConstellationIndex& self,
                         const std::vector<ConstellationNode>& nodes,
                         std::optional<Vec3> gravity) -> py::object {
            auto r = self.query(nodes, gravity);
            if (!r) return py::none();
            return py::cast(std::move(r).value());
        }, py::arg("nodes"), py::arg("gravity") = py::none())
        .def("query_all", [](const ConstellationIndex& self,
                             const std::vector<ConstellationNode>& nodes,
                             std::optional<Vec3> gravity) {
            return self.queryAll(nodes, gravity);
        }, py::arg("nodes"), py::arg("gravity") = py::none())
        // 어떤 토큰이 장소를 정의하는가를 정하는 필터. isStableLandmark 의
        // 네 조건과 sigma 오름차순 절단이 여기서만 함께 돈다.
        .def_static("build_from", &ConstellationIndex::buildFrom,
                    py::arg("tokens"), py::arg("reference"), py::arg("max_nodes") = 40)
        .def_property_readonly("place_count", &ConstellationIndex::placeCount);

    // --- 신뢰도 -----------------------------------------------------------
    py::class_<ConfidenceConfig>(m, "ConfidenceConfig")
        .def(py::init<>())
        .def_readwrite("p_detect_visible", &ConfidenceConfig::p_detect_visible)
        .def_readwrite("p_false_alarm", &ConfidenceConfig::p_false_alarm)
        .def_readwrite("logodds_min", &ConfidenceConfig::logodds_min)
        .def_readwrite("logodds_max", &ConfidenceConfig::logodds_max)
        .def_readwrite("evidence_gain", &ConfidenceConfig::evidence_gain);

    m.def("to_logodds", &ConfidenceEngine::toLogOdds);
    m.def("to_probability", &ConfidenceEngine::toProbability);

    // 토큰 전체를 노출하지 않고 믿음만 다루는 얇은 뷰.
    // 차등 테스트에 필요한 것은 이것뿐이다.
    py::enum_<TokenLifecycle>(m, "TokenLifecycle")
        .value("Provisional", TokenLifecycle::Provisional)
        .value("Active", TokenLifecycle::Active)
        .value("Occluded", TokenLifecycle::Occluded)
        .value("Dormant", TokenLifecycle::Dormant)
        .value("Displaced", TokenLifecycle::Displaced)
        .value("Retired", TokenLifecycle::Retired);

    py::class_<WorldToken, std::shared_ptr<WorldToken>>(m, "WorldToken")
        .def(py::init<>())
        .def_readwrite("existence_belief", &WorldToken::existence_belief)
        .def_readwrite("identity_belief", &WorldToken::identity_belief)
        .def_readwrite("static_belief", &WorldToken::static_belief)
        // static_prior 없이는 decayStaticBelief 가 무엇으로 돌아가는지 고를 수
        // 없어 감쇠가 항상 0.5 로만 수렴한다 - 상수 두 개를 비교하게 된다.
        .def_readwrite("static_prior", &WorldToken::static_prior)
        // 판정 척도. 이것이 없으면 updateStaticBelief 의 sigma_d 가 항상
        // motion_noise_floor 로 고정되고, 그 경로만 비교하게 된다.
        .def_readwrite("meas_sigma", &WorldToken::meas_sigma)
        .def_readwrite("miss_count", &WorldToken::miss_count)
        .def_readwrite("observation_count", &WorldToken::observation_count)
        .def_readwrite("class_id", &WorldToken::class_id)
        .def_readwrite("class_name", &WorldToken::class_name)
        .def_readwrite("lifecycle", &WorldToken::lifecycle)
        .def_readwrite("position", &WorldToken::position)
        .def_readwrite("position_cov", &WorldToken::position_cov)
        .def_readwrite("velocity", &WorldToken::velocity)
        .def_readwrite("extent", &WorldToken::extent)
        .def_property_readonly("token_id", [](const WorldToken& t) { return t.id.value; })
        .def_property("first_seen",
            [](const WorldToken& t) { return t.first_seen.seconds(); },
            [](WorldToken& t, double s) { t.first_seen = Timestamp::fromSeconds(s); })
        .def_property("last_seen",
            [](const WorldToken& t) { return t.last_seen.seconds(); },
            [](WorldToken& t, double s) { t.last_seen = Timestamp::fromSeconds(s); })
        .def_property_readonly("static_updates",
            [](const WorldToken& t) { return t.static_diag.updates; })
        .def_property_readonly("static_ratio",
            [](const WorldToken& t) { return t.static_diag.ratio; })
        .def_property_readonly("static_sigma",
            [](const WorldToken& t) { return t.static_diag.sigma; })
        .def_property("is_agent",
            [](const WorldToken& t) { return has(t.affordance, Affordance::Agent); },
            [](WorldToken& t, bool v) {
                t.affordance = v ? (t.affordance | Affordance::Agent) : t.affordance;
            })
        .def_property_readonly("box", [](const WorldToken& t) {
            return py::make_tuple(t.box.x, t.box.y, t.box.width, t.box.height);
        })
        .def("position_sigma", &WorldToken::positionSigma)
        .def("is_dynamic", &WorldToken::isDynamic)
        .def("is_alive", &WorldToken::isAlive)
        .def("is_stable_landmark", &WorldToken::isStableLandmark);

    py::class_<ConfidenceEngine>(m, "ConfidenceEngine")
        .def(py::init<ConfidenceConfig>(), py::arg("config") = ConfidenceConfig{})
        // C++ 은 env.sensor_reliability 와 obs.sensor_reliability 를 *곱한다*.
        // 예전 바인딩은 파이썬 인자 하나를 두 자리에 다 넣어 신뢰도를 제곱했고,
        // 참조 구현이 같은 값을 두 번 받도록 맞춰져 있어 차등 테스트가 그 사실을
        // 볼 수 없었다 - 두 인자가 항상 같으면 곱인지 아닌지 구분되지 않는다.
        // 이제 따로 받는다. 기본값 1.0 은 "이 관측 자체에는 감점이 없다".
        .def("on_observed", [](const ConfidenceEngine& self, WorldToken& t,
                               double detection_conf, double sensor_reliability,
                               double image_quality, double assoc_margin,
                               double obs_sensor_reliability) {
            Observation obs;
            obs.stamp = Timestamp::fromSeconds(1.0);
            obs.detection_conf = static_cast<float>(detection_conf);
            obs.sensor_reliability = obs_sensor_reliability;
            obs.image_quality = image_quality;
            EnvironmentState env;
            env.sensor_reliability = sensor_reliability;
            self.onObserved(t, obs, env, assoc_margin);
        }, py::arg("token"), py::arg("detection_conf"), py::arg("sensor_reliability") = 1.0,
           py::arg("image_quality") = 1.0, py::arg("assoc_margin") = -1.0,
           py::arg("obs_sensor_reliability") = 1.0)
        .def("on_missed", [](const ConfidenceEngine& self, WorldToken& t,
                             double expected_visibility, double sensor_reliability) {
            EnvironmentState env;
            env.sensor_reliability = sensor_reliability;
            self.onMissed(t, expected_visibility, env);
        }, py::arg("token"), py::arg("expected_visibility"),
           py::arg("sensor_reliability") = 1.0)
        .def("on_out_of_view", &ConfidenceEngine::onOutOfView)
        .def("update_static", [](const ConfidenceEngine& self, WorldToken& t,
                                 const Vec3& displacement, double dt,
                                 double sensor_reliability) {
            EnvironmentState env;
            env.sensor_reliability = sensor_reliability;
            self.updateStaticBelief(t, displacement, dt, env);
        }, py::arg("token"), py::arg("displacement"), py::arg("dt"),
           py::arg("sensor_reliability") = 1.0)
        // 자율 개체의 정적 주장 유효기간. 상한 대신 이 경로가 그 역할을 한다.
        .def("decay_static_belief", &ConfidenceEngine::decayStaticBelief,
             py::arg("token"), py::arg("dt"))
        // 루프 클로저 후 믿음 결합. 이 엔진의 설정을 쓴다 (static 메서드지만
        // 설정을 인자로 받으므로 인스턴스에 붙여야 파이썬 쪽과 대응이 맞는다).
        .def("merge_beliefs", [](const ConfidenceEngine& self, WorldToken& keep,
                                 const WorldToken& absorbed) {
            ConfidenceEngine::mergeBeliefs(keep, absorbed, self.config());
        }, py::arg("keep"), py::arg("absorbed"));

    // --- 환경 / 품질 -------------------------------------------------------
    py::class_<EnvironmentEvidence>(m, "EnvironmentEvidence")
        .def(py::init<>())
        .def_readwrite("darkness", &EnvironmentEvidence::darkness)
        .def_readwrite("haze", &EnvironmentEvidence::haze)
        .def_readwrite("rain_streak", &EnvironmentEvidence::rain_streak)
        .def_readwrite("snow_particle", &EnvironmentEvidence::snow_particle)
        .def_readwrite("dust", &EnvironmentEvidence::dust)
        .def_readwrite("motion_blur", &EnvironmentEvidence::motion_blur)
        .def_readwrite("lens_dirt", &EnvironmentEvidence::lens_dirt)
        .def_readwrite("water_drop", &EnvironmentEvidence::water_drop)
        .def_readwrite("camera_shake", &EnvironmentEvidence::camera_shake)
        .def_readwrite("noise", &EnvironmentEvidence::noise)
        .def_readwrite("texture_poverty", &EnvironmentEvidence::texture_poverty)
        // 아래 셋은 derive_adaptation 경로에서 안 쓰여 오래 안 열려 있었다.
        // 영상->증거 차분 테스트가 확인해야 하는 채널이 정확히 이쪽이다.
        .def_readwrite("backlight", &EnvironmentEvidence::backlight)
        .def_readwrite("specular", &EnvironmentEvidence::specular)
        .def_readwrite("shadow_strength", &EnvironmentEvidence::shadow_strength);

    py::class_<EnvironmentState::TierWeights>(m, "TierWeights")
        .def_readonly("photometric", &EnvironmentState::TierWeights::photometric)
        .def_readonly("constellation", &EnvironmentState::TierWeights::constellation)
        .def_readonly("structural", &EnvironmentState::TierWeights::structural)
        .def_readonly("motion_prior", &EnvironmentState::TierWeights::motion_prior);

    py::class_<EnvironmentState>(m, "EnvironmentState")
        .def(py::init<>())
        .def_readonly("evidence", &EnvironmentState::evidence)
        .def_readonly("tier", &EnvironmentState::tier)
        .def_readonly("visibility", &EnvironmentState::visibility)
        .def_readonly("camera_health", &EnvironmentState::camera_health)
        // 아래 셋은 쓰기도 연다. TokenStore 의 생애주기 타임아웃과 믿음 감쇠가
        // 이 값들에 곱해지므로, 고정하면 그 경로를 값 하나로만 밟게 된다.
        .def_readwrite("sensor_reliability", &EnvironmentState::sensor_reliability)
        .def_readwrite("memory_retention_scale", &EnvironmentState::memory_retention_scale)
        .def_readwrite("track_persistence_scale", &EnvironmentState::track_persistence_scale)
        .def_readonly("detection_threshold_scale", &EnvironmentState::detection_threshold_scale)
        .def_readonly("scene_complexity", &EnvironmentState::scene_complexity)
        .def_property_readonly("lighting", [](const EnvironmentState& s) {
            return std::string(toString(s.lighting)); })
        .def_property_readonly("weather", [](const EnvironmentState& s) {
            return std::string(toString(s.weather)); })
        .def_property_readonly("scene", [](const EnvironmentState& s) {
            return std::string(toString(s.scene)); })
        .def("summary", &EnvironmentState::summary);

    // 증거 -> 적응 매핑만 따로 호출한다 (영상 없이 차등 테스트하기 위해)
    m.def("derive_adaptation", &deriveAdaptationFrom, py::arg("evidence"),
          "EnvironmentEvidence 로부터 tier 가중치와 시간 정책을 계산한다");

    py::class_<ImageQuality>(m, "ImageQuality")
        // EnvironmentAnalyzer 가 소비하는 다섯 필드는 쓰기를 연다. 이 값들을
        // 고정하지 못하면 영상->증거 차분이 품질엔진의 오차까지 같이 재게 된다.
        .def(py::init<>())
        .def_readonly("score", &ImageQuality::score)
        .def_readonly("sharpness", &ImageQuality::sharpness)
        .def_readonly("exposure", &ImageQuality::exposure)
        .def_readwrite("brightness", &ImageQuality::brightness)
        .def_readonly("contrast", &ImageQuality::contrast)
        .def_readwrite("noise_sigma", &ImageQuality::noise_sigma)
        .def_readonly("blur_extent_px", &ImageQuality::blur_extent_px)
        .def_readwrite("noise_free", &ImageQuality::noise_free)
        .def_readwrite("blur_free", &ImageQuality::blur_free)
        .def_readwrite("occlusion_free", &ImageQuality::occlusion_free)
        .def("photometric_variance", &ImageQuality::photometricVariance)
        .def_property_readonly("weight_map",
            [](const ImageQuality& q) { return fromMat32F(q.weight_map); });

    // 설정 타입은 그것을 기본인자로 쓰는 클래스보다 먼저 등록해야 한다.
    // pybind 는 기본인자를 등록 시점에 즉시 변환하므로, 순서가 뒤집히면
    // 컴파일은 되고 import 에서 터진다 - 그리고 그 ImportError 는 삼켜진다.
    py::class_<ImageQualityConfig>(m, "ImageQualityConfig")
        .def(py::init<>())
        .def_readwrite("analysis_width", &ImageQualityConfig::analysis_width);

    py::class_<ImageQualityEngine>(m, "ImageQualityEngine")
        .def(py::init<ImageQualityConfig>(), py::arg("config") = ImageQualityConfig{})
        .def("evaluate", [](ImageQualityEngine& self, const py::array& image, double stamp) {
            Frame f;
            f.stamp = Timestamp::fromSeconds(stamp);
            cv::Mat m2 = toMat(image);
            if (m2.channels() == 3) { f.rgb = m2; cv::cvtColor(m2, f.gray, cv::COLOR_BGR2GRAY); }
            else                    { f.gray = m2; }
            f.intrinsics.fx = f.intrinsics.fy = m2.cols * 0.9;
            f.intrinsics.cx = m2.cols * 0.5;
            f.intrinsics.cy = m2.rows * 0.5;
            f.intrinsics.width = m2.cols;
            f.intrinsics.height = m2.rows;
            return self.evaluate(f);
        }, py::arg("image"), py::arg("stamp") = 1.0);

    py::class_<EnvironmentConfig>(m, "EnvironmentConfig")
        .def(py::init<>())
        .def_readwrite("analysis_width", &EnvironmentConfig::analysis_width)
        .def_readwrite("history_size", &EnvironmentConfig::history_size)
        .def_readwrite("update_hz", &EnvironmentConfig::update_hz)
        .def_readwrite("evidence_ema", &EnvironmentConfig::evidence_ema)
        .def_readwrite("dcp_patch", &EnvironmentConfig::dcp_patch)
        .def_readwrite("dark_brightness", &EnvironmentConfig::dark_brightness)
        .def_readwrite("texture_min_gradient", &EnvironmentConfig::texture_min_gradient);

    // 추정기들은 private 이다. 그대로 두고 공개 경로(update)로 비교한다 -
    // evidence_ema=1.0 이면 EMA 가 통과가 되어 그 프레임의 원 추정값이 그대로
    // evidence 에 실린다. private 을 열면 실제로 쓰이는 경로가 아닌 것을 재게 된다.
    py::class_<EnvironmentAnalyzer>(m, "EnvironmentAnalyzer")
        .def(py::init<EnvironmentConfig>(), py::arg("config") = EnvironmentConfig{})
        .def("update", [](EnvironmentAnalyzer& self, const py::array& image,
                          const ImageQuality& quality, double stamp) {
            Frame f;
            f.stamp = Timestamp::fromSeconds(stamp);
            cv::Mat m2 = toMat(image);
            if (m2.channels() == 3) { f.rgb = m2; cv::cvtColor(m2, f.gray, cv::COLOR_BGR2GRAY); }
            else                    { f.gray = m2; }
            f.intrinsics.width  = m2.cols;
            f.intrinsics.height = m2.rows;
            return self.update(f, quality);
        }, py::arg("image"), py::arg("quality"), py::arg("stamp") = 1.0,
           py::return_value_policy::copy)
        .def("set_dynamic_level", &EnvironmentAnalyzer::setDynamicLevel, py::arg("ratio"))
        .def("reset", &EnvironmentAnalyzer::reset)
        .def_property_readonly("state", &EnvironmentAnalyzer::state,
                               py::return_value_policy::copy);

    // --- 스테레오 깊이 ------------------------------------------------------
    py::class_<StereoDepthConfig>(m, "StereoDepthConfig")
        .def(py::init<>())
        .def_readwrite("min_disparity", &StereoDepthConfig::min_disparity)
        .def_readwrite("num_disparities", &StereoDepthConfig::num_disparities)
        .def_readwrite("block_size", &StereoDepthConfig::block_size)
        .def_readwrite("p1_factor", &StereoDepthConfig::p1_factor)
        .def_readwrite("p2_factor", &StereoDepthConfig::p2_factor)
        .def_readwrite("disp12_max_diff", &StereoDepthConfig::disp12_max_diff)
        .def_readwrite("pre_filter_cap", &StereoDepthConfig::pre_filter_cap)
        .def_readwrite("uniqueness_ratio", &StereoDepthConfig::uniqueness_ratio)
        .def_readwrite("speckle_window_size", &StereoDepthConfig::speckle_window_size)
        .def_readwrite("speckle_range", &StereoDepthConfig::speckle_range)
        .def_readwrite("mode_hh", &StereoDepthConfig::mode_hh)
        .def_readwrite("baseline_m", &StereoDepthConfig::baseline_m)
        .def_readwrite("focal_px", &StereoDepthConfig::focal_px)
        .def_readwrite("min_depth_m", &StereoDepthConfig::min_depth_m)
        .def_readwrite("max_depth_m", &StereoDepthConfig::max_depth_m)
        .def_readwrite("disparity_noise_px", &StereoDepthConfig::disparity_noise_px)
        .def_readwrite("max_depth_sigma_m", &StereoDepthConfig::max_depth_sigma_m);

    py::class_<StereoDepthResult>(m, "StereoDepthResult")
        .def_property_readonly("depth", [](const StereoDepthResult& r) {
            return fromMat32F(r.depth); })
        .def_property_readonly("disparity", [](const StereoDepthResult& r) {
            return fromMat32F(r.disparity); })
        .def_readonly("valid_ratio", &StereoDepthResult::valid_ratio)
        .def_readonly("median_depth_m", &StereoDepthResult::median_depth_m)
        .def_readonly("min_valid_disparity", &StereoDepthResult::min_valid_disparity)
        .def_readonly("min_representable_depth_m",
                      &StereoDepthResult::min_representable_depth_m)
        .def_readonly("clipped_ratio", &StereoDepthResult::clipped_ratio);

    py::class_<StereoDepth>(m, "StereoDepth")
        .def(py::init<StereoDepthConfig>(), py::arg("config") = StereoDepthConfig{})
        .def("compute", [](StereoDepth& self, const py::array& left, const py::array& right) {
            cv::Mat l = toMat(left), r = toMat(right);
            return self.compute(l, r);
        }, py::arg("left"), py::arg("right"))
        .def_static("min_valid_disparity", &StereoDepth::minValidDisparity, py::arg("config"))
        .def_static("required_disparities", &StereoDepth::requiredDisparities,
                    py::arg("focal_px"), py::arg("baseline_m"), py::arg("min_depth_m"));

    // --- 직접정렬 ---------------------------------------------------------
    py::enum_<InformationModel>(m, "InformationModel")
        .value("SensorVariance", InformationModel::SensorVariance)
        .value("ResidualVariance", InformationModel::ResidualVariance)
        .value("ClusterRobust", InformationModel::ClusterRobust)
        .value("EffectiveSample", InformationModel::EffectiveSample)
        .value("CoherentFrame", InformationModel::CoherentFrame);

    // --- SPA (Tier 2) ------------------------------------------------------
    // 26 절이 "PoseFusion 은 커버리지가 생겼지만 SPA / PlaneExtractor 는 여전히
    // 0" 으로 남겨 둔 부분이다. 7.1 의 결함(회전 정보행렬이 진리의 직교여집합
    // 이었다)이 정확히 이 코드에서 나왔고, 그때는 오라클이 없어서 유한차분으로
    // 잡아야 했다.
    py::class_<Plane>(m, "Plane")
        .def(py::init([](const Vec3& n, double d, std::size_t inl,
                         const Vec3& c, double ext, double rms) {
            Plane p;
            p.normal = n.normalized();
            p.distance = d;
            p.inliers = inl;
            p.centroid = c;
            p.extent = ext;
            p.rms = rms;
            return p;
        }), py::arg("normal"), py::arg("distance"), py::arg("inliers") = 100,
            py::arg("centroid") = Vec3::Zero(), py::arg("extent") = 0.5,
            py::arg("rms") = 0.0)
        .def_readwrite("normal", &Plane::normal)
        .def_readwrite("distance", &Plane::distance)
        .def_readwrite("inliers", &Plane::inliers)
        .def_readwrite("centroid", &Plane::centroid)
        .def_readwrite("extent", &Plane::extent)
        .def_readwrite("rms", &Plane::rms)
        .def_property_readonly("confidence", &Plane::confidence)
        .def("signed_distance", &Plane::signedDistance, py::arg("point"));

    py::class_<PlaneExtractorConfig>(m, "PlaneExtractorConfig")
        .def(py::init<>())
        .def_readwrite("stride", &PlaneExtractorConfig::stride)
        .def_readwrite("min_depth", &PlaneExtractorConfig::min_depth)
        .def_readwrite("max_depth", &PlaneExtractorConfig::max_depth)
        .def_readwrite("normal_bins", &PlaneExtractorConfig::normal_bins)
        .def_readwrite("distance_bin", &PlaneExtractorConfig::distance_bin)
        .def_readwrite("min_inliers", &PlaneExtractorConfig::min_inliers)
        .def_readwrite("refine_threshold", &PlaneExtractorConfig::refine_threshold)
        .def_readwrite("refine_iterations", &PlaneExtractorConfig::refine_iterations)
        .def_readwrite("max_planes", &PlaneExtractorConfig::max_planes)
        .def_readwrite("depth_discontinuity", &PlaneExtractorConfig::depth_discontinuity)
        .def_readwrite("planarity_ratio", &PlaneExtractorConfig::planarity_ratio);

    py::class_<PlaneExtractor>(m, "PlaneExtractor")
        .def(py::init<PlaneExtractorConfig>(), py::arg("config") = PlaneExtractorConfig{})
        .def("extract", [](PlaneExtractor& self, const py::array& depth,
                           double fx, double fy, double cx, double cy) {
            CameraIntrinsics K;
            K.fx = fx; K.fy = fy; K.cx = cx; K.cy = cy;
            cv::Mat d = toMat(depth);
            K.width = d.cols; K.height = d.rows;
            return unwrap(self.extract(d, K));
        }, py::arg("depth"), py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"));

    py::class_<PlaneMatch>(m, "PlaneMatch")
        .def_readonly("ref_index", &PlaneMatch::ref_index)
        .def_readonly("cur_index", &PlaneMatch::cur_index)
        .def_readonly("angle", &PlaneMatch::angle)
        .def_readonly("distance_diff", &PlaneMatch::distance_diff)
        .def_readonly("weight", &PlaneMatch::weight);

    py::class_<StructuralAlignmentResult>(m, "StructuralAlignmentResult")
        .def_readonly("T_cur_ref", &StructuralAlignmentResult::T_cur_ref)
        .def_readonly("information", &StructuralAlignmentResult::information)
        .def_readonly("eigenvalues", &StructuralAlignmentResult::eigenvalues)
        .def_readonly("weakest_direction", &StructuralAlignmentResult::weakest_direction)
        .def_readonly("observable_dof", &StructuralAlignmentResult::observable_dof)
        .def_readonly("rotation_rank", &StructuralAlignmentResult::rotation_rank)
        .def_readonly("translation_rank", &StructuralAlignmentResult::translation_rank)
        .def_readonly("normal_rms", &StructuralAlignmentResult::normal_rms)
        .def_readonly("offset_rms", &StructuralAlignmentResult::offset_rms)
        .def_readonly("matches", &StructuralAlignmentResult::matches);

    py::class_<StructuralAlignerConfig>(m, "StructuralAlignerConfig")
        .def(py::init<>());

    py::class_<StructuralAligner>(m, "StructuralAligner")
        .def(py::init<StructuralAlignerConfig>(),
             py::arg("config") = StructuralAlignerConfig{})
        .def("match", &StructuralAligner::match, py::arg("reference"), py::arg("current"),
             py::arg("init") = SE3::identity())
        .def("align", [](StructuralAligner& self, const std::vector<Plane>& ref,
                         const std::vector<Plane>& cur, const SE3& init,
                         double alpha) -> py::tuple {
            const auto r = self.align(ref, cur, init, alpha);
            // 랭크 부족은 degraded 이지 실패가 아니다. ok() 로 접어 버리면
            // "관측 안 된 축이 있다" 와 "못 풀었다" 가 같은 모양이 된다.
            if (!r.ok()) return py::make_tuple(false, py::object(py::none()));
            return py::make_tuple(true, py::cast(r.value()));
        }, py::arg("reference"), py::arg("current"),
           py::arg("init") = SE3::identity(), py::arg("alpha_structural") = 1.0);

    m.def("unobservable_directions", &unobservableDirections,
          py::arg("result"), py::arg("ratio") = 1e-3);

    // --- fusion (Tier 융합) -------------------------------------------------
    // 06-results.md 26 절이 "PoseFusion / SPA / TierInformation 은 바인딩에 아예
    // 없어 차분 커버리지가 0" 이라고 적어 둔 그 공백이다. 18절과 21절, 23.4 가
    // 전부 이 코드 위에 서 있는데 오라클이 없었다.
    py::enum_<fusion::Tier>(m, "Tier")
        .value("Photometric", fusion::Tier::Photometric)
        .value("Constellation", fusion::Tier::Constellation)
        .value("Structural", fusion::Tier::Structural);

    py::enum_<fusion::Abstain>(m, "Abstain")
        .value("None", fusion::Abstain::None)
        .value("Disabled", fusion::Abstain::Disabled)
        .value("NoInput", fusion::Abstain::NoInput)
        .value("Failed", fusion::Abstain::Failed)
        .value("ZeroInformation", fusion::Abstain::ZeroInformation)
        .value("ZeroWeight", fusion::Abstain::ZeroWeight);

    py::class_<fusion::TierEstimate>(m, "TierEstimate")
        .def(py::init<>())
        .def_readwrite("tier", &fusion::TierEstimate::tier)
        .def_readwrite("T_cur_ref", &fusion::TierEstimate::T_cur_ref)
        .def_readwrite("information", &fusion::TierEstimate::information)
        .def_readwrite("alpha", &fusion::TierEstimate::alpha)
        .def_readwrite("calibration", &fusion::TierEstimate::calibration)
        .def_readwrite("available", &fusion::TierEstimate::available)
        .def_readwrite("reason", &fusion::TierEstimate::reason);

    py::class_<fusion::TierContribution>(m, "TierContribution")
        .def_readonly("used", &fusion::TierContribution::used)
        .def_readonly("reason", &fusion::TierContribution::reason)
        .def_readonly("alpha", &fusion::TierContribution::alpha)
        .def_readonly("info_trace", &fusion::TierContribution::info_trace)
        .def_readonly("info_share", &fusion::TierContribution::info_share)
        .def_readonly("residual", &fusion::TierContribution::residual)
        .def_readonly("self_nees", &fusion::TierContribution::self_nees);

    py::class_<fusion::FusionResult>(m, "FusionResult")
        .def_readonly("T_cur_ref", &fusion::FusionResult::T_cur_ref)
        .def_readonly("information", &fusion::FusionResult::information)
        .def_readonly("information_naive", &fusion::FusionResult::information_naive)
        .def_readonly("eigenvalues", &fusion::FusionResult::eigenvalues)
        .def_readonly("weakest_direction", &fusion::FusionResult::weakest_direction)
        .def_readonly("observable_dof", &fusion::FusionResult::observable_dof)
        .def_readonly("iterations", &fusion::FusionResult::iterations)
        .def_readonly("contributing_tiers", &fusion::FusionResult::contributing_tiers)
        .def_property_readonly("tiers", [](const fusion::FusionResult& r) {
            return std::vector<fusion::TierContribution>(r.tiers.begin(), r.tiers.end());
        });

    py::class_<fusion::FusionConfig>(m, "FusionConfig")
        .def(py::init<>())
        .def_readwrite("max_iterations", &fusion::FusionConfig::max_iterations)
        .def_readwrite("convergence_delta", &fusion::FusionConfig::convergence_delta)
        .def_readwrite("degeneracy_ratio", &fusion::FusionConfig::degeneracy_ratio)
        .def_readwrite("update_observable_ratio",
                       &fusion::FusionConfig::update_observable_ratio);

    // 기권은 예외가 아니라 결과다. 파이썬에는 (ok, value) 로 준다 - 예외로 바꾸면
    // "아무 tier 도 기여하지 못했다" 와 "터졌다" 가 같은 모양이 된다.
    m.def("fuse", [](const std::vector<fusion::TierEstimate>& es,
                     fusion::FusionConfig cfg) -> py::tuple {
        const auto r = fusion::fuse(es, cfg);
        if (!r.ok()) return py::make_tuple(false, py::object(py::none()));
        return py::make_tuple(true, py::cast(r.value()));
    }, py::arg("estimates"), py::arg("config") = fusion::FusionConfig{});

    m.def("se3_left_jacobian", &fusion::se3LeftJacobian, py::arg("xi"));
    m.def("se3_left_jacobian_inverse", &fusion::se3LeftJacobianInverse, py::arg("xi"));
    m.def("transport_information", &fusion::transportInformation,
          py::arg("information"), py::arg("T_from"), py::arg("T_to"));

    py::class_<AlignmentResult>(m, "AlignmentResult")
        .def_readonly("T_cur_ref", &AlignmentResult::T_cur_ref)
        .def_readonly("information", &AlignmentResult::information)
        .def_readonly("eigenvalues", &AlignmentResult::eigenvalues)
        .def_readonly("observable_dof", &AlignmentResult::observable_dof)
        .def_readonly("affine_a", &AlignmentResult::affine_a)
        .def_readonly("affine_b", &AlignmentResult::affine_b)
        .def_readonly("photometric_rmse", &AlignmentResult::photometric_rmse)
        .def_readonly("inlier_ratio", &AlignmentResult::inlier_ratio)
        .def_readonly("point_count", &AlignmentResult::point_count)
        .def_readonly("iterations", &AlignmentResult::iterations)
        // 25절의 독립 실패 신호. 노출하지 않으면 파이썬 쪽에서 검증할 수 없다.
        .def_readonly("depth_consistency", &AlignmentResult::depth_consistency)
        .def_readonly("depth_outlier_ratio", &AlignmentResult::depth_outlier_ratio)
        // 기본값을 여기 다시 적지 않는다. 숫자를 복사해 두면 헤더에서 보정된
        // 상수를 고쳐도 파이썬 쪽은 옛 값을 계속 쓰고, 두 언어가 다른 문턱으로
        // 같은 이름의 판정을 하게 된다.
        .def("depth_consistent", &AlignmentResult::depthConsistent,
             py::arg("max_rel") = AlignmentResult::kDepthConsistencyGate)
        .def("depth_reliability", &AlignmentResult::depthReliability,
             py::arg("max_rel") = AlignmentResult::kDepthConsistencyGate)
        .def_readonly_static("DEPTH_GATE", &AlignmentResult::kDepthConsistencyGate);

    py::class_<DirectAlignerConfig>(m, "DirectAlignerConfig")
        .def(py::init<>())
        .def_readwrite("pyramid_levels", &DirectAlignerConfig::pyramid_levels)
        .def_readwrite("max_iterations", &DirectAlignerConfig::max_iterations)
        .def_readwrite("grid_cell", &DirectAlignerConfig::grid_cell)
        // huber_delta 는 적응형 커널로 교체되면서 사라졌다. 이 줄이 남아 바인딩
        // 전체가 컴파일되지 않았고, 그 결과 차등 테스트 41 개가 통째로 skip 되면서
        // 초록으로 보였다. 설정 필드를 지울 때는 반드시 여기도 같이 본다.
        .def_readwrite("huber_k", &DirectAlignerConfig::huber_k)
        .def_readwrite("huber_noise_ratio", &DirectAlignerConfig::huber_noise_ratio)
        .def_readwrite("huber_min_delta", &DirectAlignerConfig::huber_min_delta)
        .def_readwrite("information_model", &DirectAlignerConfig::information_model)
        .def_readwrite("effective_samples", &DirectAlignerConfig::effective_samples)
        .def_readwrite("coherent_sigma", &DirectAlignerConfig::coherent_sigma)
        .def_readwrite("min_gradient", &DirectAlignerConfig::min_gradient)
        .def_readwrite("degeneracy_ratio", &DirectAlignerConfig::degeneracy_ratio)
        .def_readwrite("level_observable_ratio", &DirectAlignerConfig::level_observable_ratio)
        .def_readwrite("affine_prior_weight", &DirectAlignerConfig::affine_prior_weight)
        .def_readwrite("estimate_affine", &DirectAlignerConfig::estimate_affine)
        .def_readwrite("min_depth", &DirectAlignerConfig::min_depth)
        .def_readwrite("max_depth", &DirectAlignerConfig::max_depth)
        .def_readwrite("depth_sigma_rel", &DirectAlignerConfig::depth_sigma_rel);

    py::class_<DirectAligner>(m, "DirectAligner")
        .def(py::init([](DirectAlignerConfig cfg) {
            return std::make_unique<DirectAligner>(cfg, nullptr);
        }), py::arg("config") = DirectAlignerConfig{})
        .def("align", [](DirectAligner& self,
                         const py::array& ref_gray, const py::array& ref_depth,
                         const py::array& cur_gray,
                         double fx, double fy, double cx, double cy,
                         const SE3& init, std::optional<py::array> static_mask,
                         std::optional<py::array> cur_depth) {
            Frame ref, cur;
            ref.gray = toMat(ref_gray);
            ref.depth = toMat(ref_depth);
            cur.gray = toMat(cur_gray);
            // cur 깊이는 정렬에 쓰이지 않는다. 25절의 기하 정합성 신호를
            // 파이썬에서 검증할 수 있게 하려고 받는다 - 없으면 그 값은
            // 언제나 -1(판정 불가)이라 테스트가 아무 것도 재지 못한다.
            if (cur_depth) cur.depth = toMat(*cur_depth);
            if (static_mask) ref.static_mask = toMat(*static_mask);

            CameraIntrinsics K;
            K.fx = fx; K.fy = fy; K.cx = cx; K.cy = cy;
            K.width = ref.gray.cols; K.height = ref.gray.rows;
            ref.intrinsics = cur.intrinsics = K;
            ref.stamp = Timestamp::fromSeconds(1.0);
            cur.stamp = Timestamp::fromSeconds(1.033);
            ref.sensor = SensorKind::RgbD;
            cur.sensor = cur.depth.empty() ? SensorKind::Monocular : SensorKind::RgbD;

            return unwrap(self.align(ref, cur, init));
        }, py::arg("ref_gray"), py::arg("ref_depth"), py::arg("cur_gray"),
           py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
           py::arg("init") = SE3::identity(), py::arg("static_mask") = py::none(),
           py::arg("cur_depth") = py::none());

    // --- 검출 후처리 -------------------------------------------------------
    py::class_<Detection>(m, "Detection")
        .def(py::init([](int cls, const std::string& name,
                         float x, float y, float w, float h, float conf) {
            Detection d;
            d.class_id = cls;
            d.class_name = name;
            d.box = cv::Rect2f(x, y, w, h);
            d.confidence = conf;
            return d;
        }))
        .def_readwrite("class_id", &Detection::class_id)
        .def_readwrite("class_name", &Detection::class_name)
        .def_readwrite("confidence", &Detection::confidence)
        .def_property_readonly("box", [](const Detection& d) {
            return py::make_tuple(d.box.x, d.box.y, d.box.width, d.box.height);
        });

    // --- 토큰 저장소 -------------------------------------------------------
    py::class_<TokenStoreConfig>(m, "TokenStoreConfig")
        .def(py::init<>())
        .def_readwrite("max_association_distance", &TokenStoreConfig::max_association_distance)
        .def_readwrite("max_association_mahalanobis",
                       &TokenStoreConfig::max_association_mahalanobis)
        .def_readwrite("assoc_maneuver_speed", &TokenStoreConfig::assoc_maneuver_speed)
        // assoc_pose_rot_sigma 는 17.2 의 가설이 기각되면서 사라졌다.
        // 설정 필드를 지울 때는 반드시 이 파일도 같이 본다 - 19 장이 통째로
        // 그 실수 하나에서 나왔다.
        .def_readwrite("iou_weight", &TokenStoreConfig::iou_weight)
        .def_readwrite("distance_weight", &TokenStoreConfig::distance_weight)
        .def_readwrite("allow_cross_class", &TokenStoreConfig::allow_cross_class)
        .def_readwrite("depth_noise_coeff", &TokenStoreConfig::depth_noise_coeff)
        .def_readwrite("bearing_noise_px", &TokenStoreConfig::bearing_noise_px)
        .def_readwrite("no_depth_sigma", &TokenStoreConfig::no_depth_sigma)
        .def_readwrite("depth_sample_shrink", &TokenStoreConfig::depth_sample_shrink)
        .def_readwrite("depth_centroid_offset", &TokenStoreConfig::depth_centroid_offset)
        .def_readwrite("observations_to_activate", &TokenStoreConfig::observations_to_activate)
        .def_readwrite("occluded_timeout_s", &TokenStoreConfig::occluded_timeout_s)
        .def_readwrite("dormant_timeout_s", &TokenStoreConfig::dormant_timeout_s)
        .def_readwrite("existence_retire_threshold",
                       &TokenStoreConfig::existence_retire_threshold)
        .def_readwrite("existence_displace_threshold",
                       &TokenStoreConfig::existence_displace_threshold)
        .def_readwrite("dynamic_mask_dilate", &TokenStoreConfig::dynamic_mask_dilate);

    py::class_<IntegrationReport>(m, "IntegrationReport")
        .def_readonly("matched", &IntegrationReport::matched)
        .def_readonly("created", &IntegrationReport::created)
        .def_readonly("missed", &IntegrationReport::missed)
        .def_readonly("retired", &IntegrationReport::retired)
        .def_readonly("dynamic_area_ratio", &IntegrationReport::dynamic_area_ratio)
        // 연관 실패의 원인 분해. 하나로 뭉치면 "게이트가 좁다" 와 "후보가 없다"
        // 가 구분되지 않고, 그 둘은 고칠 곳이 다르다.
        .def_readonly("det_no_candidate", &IntegrationReport::det_no_candidate)
        .def_readonly("det_gated_out", &IntegrationReport::det_gated_out)
        .def_readonly("det_unassigned", &IntegrationReport::det_unassigned)
        .def_readonly("gated_maha_sum", &IntegrationReport::gated_maha_sum)
        .def_readonly("gated_dist_sum", &IntegrationReport::gated_dist_sum);

    py::class_<TokenStore::MaskReport>(m, "MaskReport")
        .def_readonly("masked_ratio", &TokenStore::MaskReport::masked_ratio)
        .def_readonly("from_observed", &TokenStore::MaskReport::from_observed)
        .def_readonly("from_stale", &TokenStore::MaskReport::from_stale)
        .def_readonly("withheld_unjudged", &TokenStore::MaskReport::withheld_unjudged)
        .def_readonly("n_masking", &TokenStore::MaskReport::n_masking)
        .def_readonly("n_withheld", &TokenStore::MaskReport::n_withheld);

    py::class_<TokenStore>(m, "TokenStore")
        .def(py::init<TokenStoreConfig, ConfidenceConfig>(),
             py::arg("config") = TokenStoreConfig{},
             py::arg("confidence") = ConfidenceConfig{})
        // Frame 을 그대로 노출하지 않는다. TokenStore 가 실제로 읽는 것은
        // 내부파라미터와 깊이맵뿐이라, 그 둘만 인자로 받아 여기서 조립한다.
        .def("integrate", [](TokenStore& self, const std::vector<Detection>& dets,
                             double stamp, const py::array_t<double>& intrinsics,
                             std::optional<py::array> depth, const SE3& T_world_cam,
                             std::optional<EnvironmentState> env) {
            if (intrinsics.size() != 6) {
                throw std::invalid_argument("intrinsics 는 (fx, fy, cx, cy, width, height)");
            }
            const auto* k = intrinsics.data();
            Frame f;
            f.intrinsics.fx = k[0]; f.intrinsics.fy = k[1];
            f.intrinsics.cx = k[2]; f.intrinsics.cy = k[3];
            f.intrinsics.width  = static_cast<int>(k[4]);
            f.intrinsics.height = static_cast<int>(k[5]);
            f.stamp = Timestamp::fromSeconds(stamp);
            if (depth) { f.depth = toMat(*depth); f.sensor = SensorKind::RgbD; }

            DetectionSet ds;
            ds.stamp = Timestamp::fromSeconds(stamp);
            ds.items = dets;

            const EnvironmentState e = env ? *env : EnvironmentState{};
            return unwrap(self.integrate(ds, f, T_world_cam, e));
        }, py::arg("detections"), py::arg("stamp"), py::arg("intrinsics"),
           py::arg("depth") = py::none(), py::arg("T_world_cam") = SE3::identity(),
           py::arg("env") = py::none())
        .def("build_static_mask", [](const TokenStore& self,
                                     const py::array_t<double>& intrinsics,
                                     const SE3& T_world_cam) {
            if (intrinsics.size() != 6) {
                throw std::invalid_argument("intrinsics 는 (fx, fy, cx, cy, width, height)");
            }
            const auto* k = intrinsics.data();
            Frame f;
            f.intrinsics.fx = k[0]; f.intrinsics.fy = k[1];
            f.intrinsics.cx = k[2]; f.intrinsics.cy = k[3];
            f.intrinsics.width  = static_cast<int>(k[4]);
            f.intrinsics.height = static_cast<int>(k[5]);

            TokenStore::MaskReport rep;
            const cv::Mat mask = self.buildStaticMask(f, T_world_cam, &rep);
            py::array_t<std::uint8_t> out({mask.rows, mask.cols});
            if (!mask.empty()) {
                std::memcpy(out.mutable_data(), mask.data,
                            static_cast<std::size_t>(mask.total()));
            }
            return py::make_tuple(out, rep);
        }, py::arg("intrinsics"), py::arg("T_world_cam") = SE3::identity())
        .def("merge", [](TokenStore& self, std::uint64_t keep, std::uint64_t absorb) {
            const Status s = self.merge(TokenId(keep), TokenId(absorb));
            if (!s) throw std::runtime_error(s.error().message());
        }, py::arg("keep"), py::arg("absorb"))
        .def("all_tokens", &TokenStore::allTokens)
        .def("active_tokens", &TokenStore::activeTokens)
        .def("stable_landmarks", &TokenStore::stableLandmarks)
        .def("find", [](const TokenStore& self, std::uint64_t id) {
            return self.find(TokenId(id));
        }, py::arg("token_id"))
        .def("clear", &TokenStore::clear)
        .def("__len__", &TokenStore::size);

    m.def("box_iou", [](py::tuple a, py::tuple b) {
        return boxIoU(cv::Rect2f(a[0].cast<float>(), a[1].cast<float>(),
                                 a[2].cast<float>(), a[3].cast<float>()),
                      cv::Rect2f(b[0].cast<float>(), b[1].cast<float>(),
                                 b[2].cast<float>(), b[3].cast<float>()));
    });

    m.def("non_max_suppression", &nonMaxSuppression,
          py::arg("detections"), py::arg("iou_threshold") = 0.45f,
          py::arg("max_output") = 300);
}
