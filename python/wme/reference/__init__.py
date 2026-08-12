"""C++ 엔진 핵심 알고리즘의 numpy 참조 구현 (차등 테스트용)."""

from .assignment import INFEASIBLE, solve_assignment
from .confidence import Beliefs, ConfidenceConfig, ConfidenceEngine, to_logodds, to_probability
from .constellation import ConstellationIndex, Config as ConstellationConfig, Match, Node, Place
from .environment import PRESETS, Adaptation, Evidence, derive_adaptation
from .equirect import (
    PinholeView,
    build_maps,
    derive_intrinsics,
    direction_from_equirect,
    equirect_from_direction,
    max_vertical_disparity_px,
    rectifiable_yaw_limit_deg,
    view_rotation,
)
from .geometry import (
    SE3,
    kabsch,
    matrix_to_quat,
    quat_to_matrix,
    skew,
    so3_exp,
    so3_left_jacobian,
    so3_left_jacobian_inv,
    so3_log,
)
from .tokens import (
    Detection,
    Intrinsics,
    Lifecycle,
    Token,
    TokenStore,
    TokenStoreConfig,
    build_constellation_from,
)

__all__ = [
    "TokenStore", "TokenStoreConfig", "Token", "Detection", "Intrinsics",
    "Lifecycle", "build_constellation_from",
    "SE3", "skew", "so3_exp", "so3_log", "so3_left_jacobian", "so3_left_jacobian_inv",
    "kabsch", "quat_to_matrix", "matrix_to_quat",
    "solve_assignment", "INFEASIBLE",
    "ConstellationIndex", "ConstellationConfig", "Node", "Place", "Match",
    "ConfidenceEngine", "ConfidenceConfig", "Beliefs", "to_logodds", "to_probability",
    "Evidence", "Adaptation", "derive_adaptation", "PRESETS",
    # 등장방형 -> 원근. render() 는 cv2 를 쓰므로 여기서 재노출하지 않는다 -
    # 이 패키지를 import 하는 것만으로 cv2 를 요구하게 만들지 않기 위해서다.
    "PinholeView", "derive_intrinsics", "view_rotation", "build_maps",
    "direction_from_equirect", "equirect_from_direction",
    "max_vertical_disparity_px", "rectifiable_yaw_limit_deg",
]
