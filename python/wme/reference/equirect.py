"""등장방형(equirectangular) 360° 영상 -> 원근(핀홀) 뷰 잘라내기의 numpy 참조 구현.

C++ tools/equirect_convert.cpp 와 규약이 동일해야 한다. 이 모듈이 있는 이유는
편의가 아니라 차등 테스트다 - 같은 입력에 같은 출력을 내지 않으면 둘 중 하나가
틀린 것이다 (reference/geometry.py 와 같은 취지).

왜 이 단계가 필요한가
---------------------
파이프라인(StereoDepth -> DirectAligner -> tum_odometry)은 전부 **핀홀** 을
가정한다. 직선이 직선으로 오고, u = fx*X/Z + cx 가 성립한다는 가정이다.
등장방형은 그 가정을 만족하지 않는다 - 화소 좌표가 (경도, 위도)라서 직선이
휘고, 어떤 (fx, fy, cx, cy) 로도 근사되지 않는다. 그래서 360° 영상을 그대로
넣으면 실패하지 않고 **조용히 틀린다**. 잘라내는 단계가 필요한 이유가 이것이다.

좌표 규약 (여기서 한 번만 정하고 C++ 이 그대로 따른다)
--------------------------------------------------
파노라마 좌표계: X 오른쪽, Y 아래, Z 앞. 카메라 좌표계와 같은 손이다.
  경도 lam : Z 축(정면)에서 X 축(오른쪽)으로 도는 각. 범위 [-pi, pi)
  위도 phi : 적도에서 위(-Y)로 재는 각. 범위 [-pi/2, pi/2]
  방향벡터 d(lam, phi) = (cos phi * sin lam, -sin phi, cos phi * cos lam)

등장방형 화소 (u_e, v_e) 는 **화소 인덱스** 다 (정수 = 화소 중심).
가로 W_e 화소가 2pi 를, 세로 H_e 화소가 pi 를 균등 분할하고, 인덱스 u_e 화소가
덮는 구간의 **중심** 이 그 화소의 각이다:
  lam(u_e) = (u_e + 0.5) / W_e * 2pi - pi
  phi(v_e) = pi/2 - (v_e + 0.5) / H_e * pi
역변환:
  u_e(lam) = (lam + pi) / (2pi) * W_e - 0.5
  v_e(phi) = (pi/2 - phi) / pi * H_e - 0.5

이 +0.5 를 빼먹으면 어디서도 실패하지 않고 반 화소가 계속 틀린다. cv2.remap /
cv::remap 의 map 값이 **화소 인덱스** 좌표라서 (map 값 0.0 = 0번 화소의 중심),
같은 규약으로 맞춰야 한다.

규약 실데이터 검증 (2026-08-13) — **확인됨**
--------------------------------------------
위 규약에는 합성 검증으로 **원리적으로** 잡을 수 없는 자유도가 하나 있었다.
경도의 증가 방향, 즉 u 가 왼쪽에서 오른쪽으로 갈 때 시선이 오른쪽으로 도는가
왼쪽으로 도는가다. equirect_validate.py 는 이 모듈의 규약으로 원본을 만들고
같은 규약으로 되읽으므로, 규약이 통째로 뒤집혀 있어도 오차 0 으로 통과한다.
반전은 --yaw 로 흡수되지도 않는다 (원점 차이는 흡수되지만 손잡이는 아니다).
그래서 실제 스티처의 출력으로만 답할 수 있었다. 아래가 그 답이다.

표본 (둘 다 소비자 360 카메라의 **네이티브 스티치**, 서로 다른 제조사):
  A. GoPro Max     5760x2880, Main St, Oatman AZ. EXIF GPS 35.025940/-114.383136,
                   2026-04-05 22:41:04 UTC. XMP ProjectionType=equirectangular.
  B. RICOH THETA SC 5376x2688, 10th & U St NW, Washington DC. EXIF GPS
                   38.916547/-77.026062, 2017-02-18 16:29:46 UTC.
출처/라이선스는 data/equirect_sample/SOURCE.md. 둘 다 PoseHeadingDegrees 가
없어서 스티처가 스스로 방위를 말해 주지 않는다 - 그래서 태양으로 구했다.

(1) 좌우 반전 없음 — 글자.
    글자는 거울상이면 바로 보인다. 두 원본 모두, 그리고 이 규약으로 잘라낸
    원근 뷰 모두에서 글자가 정상으로 읽힌다: "SHOOTIN GALLERY", "RESTAURANT",
    "JENNY and JACKS ARTifacts", "Sugar SALOON", "SNAGGLETOOTH BURRO",
    "THE JERKY JAMBOREE" (A), "Spirit of Freedom / Ed Hamilton, Sculptor" (B).
    렌더가 원본의 좌우 순서를 보존하므로, 원본이 거울상이 아니라는 것이 곧
    u 증가 = 시계방향(북->동->남->서) 이라는 뜻이다.

(2) 절대 방위로 재확인 — 태양(계산값) + OSM(정답 방위).
    태양 위치는 GPS 와 UTC 시각에서 NOAA 알고리즘으로 계산된다. 추측이 아니다.

    A. 계산 태양 방위 246.22도, 고도 39.53도. 영상에서 잰 값 lam=+133.60도,
       phi=+38.45도 (고도가 1.08도 낮은 것은 포화 플레어 무게중심의 치우침).
       -> 영상 중앙의 방위 = 246.22 - 133.60 = 112.62도.
       길의 두 소실 방향을 지평선 띠에서 읽으면 lam = +18도 와 -162도 (대척,
       즉 곧은 길이라는 것과 앞뒤가 맞는다). 방위로 옮기면 130.6도 / 310.6도.
       OSM 의 Oatman Road(Route 66) 실제 방위는 134.1도 / 314.1도.
       **오차 3.5도, 양쪽이 같은 부호** (= 태양 무게중심의 치우침 하나로 설명된다).
       반전 가정이면 영상 중앙이 19.82도가 되고 길은 1.8도가 되어 정답에서
       132도 어긋난다. 3.5도 대 132도라 애매하지 않다.

    B. 계산 태양 방위 163.62도, 고도 38.24도. 영상에서 잰 값 lam=+158.7도,
       phi=+41.0도 (흐린 하늘이라 최대휘도점이 2.8도 벌어졌다).
       -> 영상 중앙의 방위 = 4.9도. OSM 에 등록된 Prince Hall Masonic Temple
       (큰 석조 건물)이 카메라에서 방위 288.8도, 21 m 거리에 있다. 이 규약이면
       그 건물은 lam = 288.8 - 4.9 = -76.1도 에 있어야 한다. yaw=-76.1 로
       잘라내니 실제로 그 건물이 화면을 채웠다. 반전 가정은 +33.6도 를
       가리키는데 거기는 빈 광장이다.

    두 장은 제조사·도시·연도가 모두 다르다. 같은 결론이 나왔으므로 이것은
    한 회사의 관례가 아니라 등장방형 형식 자체의 규약이다.

(3) 경도 원점 — 영상 중앙 = 카메라 정면. 지리적 고정 원점은 없다.
    위에서 나온 중앙 방위가 A 는 약 116도, B 는 약 5도로 서로 무관하다. 즉
    중앙이 북쪽이라는 식의 고정 규약은 없고, 촬영마다 카메라가 향한 쪽일
    뿐이다. Google Photo Sphere(GPano) 규격도 같은 이야기를 한다 -
    PoseHeadingDegrees 를 "the center of the image" 의 나침반 방위로 정의하고,
    (0,0,0) 이면 "the center pixel faces due north" 라고 적는다. 즉 방위의
    기준점은 언제나 **중앙 화소** 다. 그러므로 원점 차이는 --yaw 로 흡수된다.

(4) 위도 규약도 맞다. 위 (2) 에서 잰 태양의 phi 가 계산 고도와 A 는 1.08도,
    B 는 2.8도 안에서 맞았다. v=0 이 천정, phi = pi/2 - (v+0.5)/H*pi 가 맞다는
    뜻이다. 이건 좌우 반전과 무관하게 성립하는 독립된 확인이다.

(5) 이음매(lam = +-180). yaw=180 뷰(이음매가 화면 한가운데 온다)와, 원본을
    가로로 W/2 굴려서 같은 방향을 이음매 없이 렌더한 뷰를 비교하면 최대차
    5 LSB, 평균 0.0004 LSB, 1 LSB 초과 화소 0.0072%. 이음매 열의 가로기울기는
    4.13 으로 주변 20 열의 중앙값 5.01 보다 오히려 낮다 - 세로줄이 없다.
    WRAP_PAD 가 실제로 일을 하고 있다.

(6) C++ 대 이 모듈, **실데이터**. 두 파일 x 7 개 뷰(yaw 0/90/180/270,
    pitch 10 포함)에서 최대 화소차 4 LSB, 평균 0.15 LSB 이하. 합성에서 1 LSB
    였던 것이 4 LSB 로 는 것은 실사진의 고주파 성분에서 cubic 반올림이 갈리기
    때문이고, 기하가 어긋나면 이런 크기로 끝나지 않는다.

내부파라미터 유도 (임의 상수 없음)
---------------------------------
출력 폭 W, 수평 화각 hfov 가 주어졌다고 하자. 핀홀에서 화소 인덱스 u 는
  u = fx * (X/Z) + cx
이고 영상이 덮는 물리적 구간은 화소 인덱스 -0.5 (0번 화소의 왼쪽 경계) 부터
W-0.5 (W-1번 화소의 오른쪽 경계) 까지, 즉 폭 W 화소다. 시야를 좌우 대칭으로
두면
  cx = (W - 1) / 2          <- -0.5 와 W-0.5 의 중점
이고, 중심에서 경계까지의 거리는 (W-0.5) - (W-1)/2 = W/2 화소다. 그 경계가
광축과 이루는 각이 hfov/2 이므로
  tan(hfov/2) = (W/2) / fx   =>   fx = (W/2) / tan(hfov/2)
같은 논리로 세로는
  cy = (H - 1) / 2,   fy = (H/2) / tan(vfov/2)

hfov 만 주면 정사각 화소(fy = fx)를 택한다. 등장방형 원본의 화소는 적도에서
가로세로 각크기가 같으므로(2pi/W_e vs pi/H_e, 보통 W_e = 2*H_e 라 동일) 정사각이
자연스럽고, 비정사각을 쓰려면 근거가 있어야 한다. 그때 세로 화각은 유도된다:
  vfov = 2 * atan((H/2) / fy)

주의: "hfov 를 바깥쪽 화소 **중심** 사이의 각으로 정의" 하는 관례도 있고 그러면
fx = ((W-1)/2)/tan(hfov/2) 다. 둘 중 아무거나 골라도 되지만 고른 것을 K 와
일관되게 써야 한다. 여기서는 경계 기준을 쓰고, 그래야 cx = (W-1)/2 와 앞뒤가
맞는다 (위 유도 참조).

뷰 회전
-------
R_pano_cam = R_y(yaw) * R_x(pitch),  roll = 0.
  R_y(yaw)   : +Y(아래) 축 우수 회전. R_y*(0,0,1) = (sin yaw, 0, cos yaw) 이므로
               yaw 가 곧 시선 중심의 경도다.
  R_x(pitch) : +X(오른쪽) 축 우수 회전. R_x*(0,0,1) = (0, -sin pitch, cos pitch)
               이고 Y 가 아래라서 pitch 가 곧 시선 중심의 위도(양수 = 위)다.
곱 순서가 이래야 yaw 로 돌린 뒤에도 영상의 세로축이 자오선 안에 남는다 - 즉
파노라마의 "위" 가 잘라낸 뷰에서도 위다. 반대로 곱하면 roll 이 섞인다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeView:
    """잘라낸 원근 뷰의 핀홀 내부파라미터와 그 뷰가 보는 방향."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    yaw_deg: float
    pitch_deg: float
    hfov_deg: float
    vfov_deg: float

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])


def derive_intrinsics(width: int, height: int, hfov_deg: float,
                      yaw_deg: float = 0.0, pitch_deg: float = 0.0,
                      vfov_deg: float | None = None) -> PinholeView:
    """(W, H, hfov) -> (fx, fy, cx, cy). 유도는 모듈 머리말에 있다.

    vfov_deg 를 주면 fy 를 그것으로 따로 정한다(비정사각 화소). 안 주면 fy = fx.
    """
    if width < 2 or height < 2:
        raise ValueError(f"출력 해상도가 너무 작다: {width}x{height}")
    if not (0.0 < hfov_deg < 180.0):
        # 180도에서 tan 이 발산한다. 핀홀은 반구 이상을 담을 수 없다 -
        # 이건 구현 한계가 아니라 투영의 성질이다.
        raise ValueError(f"hfov 는 (0, 180) 도여야 한다: {hfov_deg}")

    hfov = np.deg2rad(hfov_deg)
    fx = (width / 2.0) / np.tan(hfov / 2.0)
    if vfov_deg is None:
        fy = fx
        vfov = 2.0 * np.arctan((height / 2.0) / fy)
    else:
        if not (0.0 < vfov_deg < 180.0):
            raise ValueError(f"vfov 는 (0, 180) 도여야 한다: {vfov_deg}")
        vfov = np.deg2rad(vfov_deg)
        fy = (height / 2.0) / np.tan(vfov / 2.0)

    return PinholeView(
        width=int(width), height=int(height),
        fx=float(fx), fy=float(fy),
        cx=(width - 1) / 2.0, cy=(height - 1) / 2.0,
        yaw_deg=float(yaw_deg), pitch_deg=float(pitch_deg),
        hfov_deg=float(hfov_deg), vfov_deg=float(np.rad2deg(vfov)),
    )


def view_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """R_pano_cam = R_y(yaw) * R_x(pitch). 머리말의 곱 순서 근거 참조."""
    y = np.deg2rad(yaw_deg)
    p = np.deg2rad(pitch_deg)
    Ry = np.array([[np.cos(y), 0.0, np.sin(y)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(y), 0.0, np.cos(y)]])
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(p), -np.sin(p)],
                   [0.0, np.sin(p), np.cos(p)]])
    return Ry @ Rx


def direction_from_equirect(u_e, v_e, src_w: int, src_h: int) -> np.ndarray:
    """등장방형 화소 인덱스 -> 단위 방향벡터 (..., 3). 머리말의 규약 그대로."""
    u_e = np.asarray(u_e, dtype=float)
    v_e = np.asarray(v_e, dtype=float)
    lam = (u_e + 0.5) / src_w * (2.0 * np.pi) - np.pi
    phi = np.pi / 2.0 - (v_e + 0.5) / src_h * np.pi
    return np.stack([np.cos(phi) * np.sin(lam),
                     -np.sin(phi),
                     np.cos(phi) * np.cos(lam)], axis=-1)


def equirect_from_direction(d: np.ndarray, src_w: int, src_h: int):
    """단위 방향벡터 -> 등장방형 화소 인덱스 (u_e, v_e). 위 함수의 역."""
    d = np.asarray(d, dtype=float)
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    d = d / np.maximum(n, 1e-300)
    lam = np.arctan2(d[..., 0], d[..., 2])
    # asin 인자를 잘라 준다. 정규화 뒤라도 부동소수 반올림으로 |y| 가 1 을
    # 아주 조금 넘을 수 있고, 그러면 asin 이 NaN 을 낸다 - 그 NaN 이 remap 까지
    # 조용히 흘러가면 검은 화소로 보일 뿐 원인이 안 보인다.
    phi = np.arcsin(np.clip(-d[..., 1], -1.0, 1.0))
    u_e = (lam + np.pi) / (2.0 * np.pi) * src_w - 0.5
    v_e = (np.pi / 2.0 - phi) / np.pi * src_h - 0.5
    return u_e, v_e


def build_maps(view: PinholeView, src_w: int, src_h: int):
    """원근 뷰 각 화소가 등장방형의 어느 화소에서 오는지 (map_x, map_y).

    반환값은 float32, 등장방형 **화소 인덱스** 좌표. map_x 는 [-0.5, src_w-0.5)
    안에 있고 보간용 여유는 호출자가 가로 방향으로 감아서(wrap) 채워야 한다.
    """
    u = np.arange(view.width, dtype=float)
    v = np.arange(view.height, dtype=float)
    uu, vv = np.meshgrid(u, v)

    # 카메라 좌표계 광선. 정규화는 아래 asin 을 위해 필요하다.
    x = (uu - view.cx) / view.fx
    y = (vv - view.cy) / view.fy
    d_cam = np.stack([x, y, np.ones_like(x)], axis=-1)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)

    R = view_rotation(view.yaw_deg, view.pitch_deg)
    d_pano = d_cam @ R.T

    map_x, map_y = equirect_from_direction(d_pano, src_w, src_h)
    return map_x.astype(np.float32), map_y.astype(np.float32)


# 가로 감기 여유. 등장방형은 경도 방향으로 이어져 있어서 lam = -pi 근처 화소의
# 보간 이웃이 반대쪽 끝에 있다. 그냥 BORDER_REPLICATE 로 두면 이음매에 세로줄이
# 생긴다. INTER_CUBIC 이 한쪽으로 2 화소, INTER_LANCZOS4 가 4 화소를 보므로
# 가장 넓은 경우의 두 배인 8 을 쓴다.
WRAP_PAD = 8


def wrap_pad(src: np.ndarray, pad: int = WRAP_PAD) -> np.ndarray:
    """등장방형 영상을 가로로 감아서 좌우에 pad 화소씩 붙인다."""
    return np.concatenate([src[:, -pad:], src, src[:, :pad]], axis=1)


def render(src: np.ndarray, view: PinholeView, interp: str = "cubic") -> np.ndarray:
    """등장방형 영상에서 원근 뷰를 잘라낸다. cv2 를 쓴다 (C++ 과 같은 remap)."""
    import cv2

    flags = {"nearest": cv2.INTER_NEAREST,
             "linear": cv2.INTER_LINEAR,
             "cubic": cv2.INTER_CUBIC,
             "lanczos4": cv2.INTER_LANCZOS4}
    if interp not in flags:
        raise ValueError(f"모르는 보간법: {interp}")

    h, w = src.shape[:2]
    map_x, map_y = build_maps(view, w, h)
    padded = wrap_pad(src)
    # 세로는 극(pole) 바깥으로 나가지 않으므로 BORDER_REPLICATE 로 충분하다.
    return cv2.remap(padded, (map_x + WRAP_PAD).astype(np.float32), map_y,
                     flags[interp], borderMode=cv2.BORDER_REPLICATE)


def project(view: PinholeView, points_pano: np.ndarray):
    """파노라마 좌표계의 3D 점 -> 원근 뷰의 화소 인덱스 (u, v, z_cam).

    검증에서 "이 3D 점은 잘라낸 영상의 어디에 있어야 하는가" 를 계산하는 데 쓴다.
    z_cam <= 0 인 점은 카메라 뒤라 화면에 없다 - 호출자가 걸러야 한다.
    """
    P = np.asarray(points_pano, dtype=float).reshape(-1, 3)
    R = view_rotation(view.yaw_deg, view.pitch_deg)
    p_cam = P @ R  # R^T @ p  (R 은 pano<-cam 이므로 cam<-pano 는 전치)
    z = p_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = view.fx * p_cam[:, 0] / z + view.cx
        v = view.fy * p_cam[:, 1] / z + view.cy
    return u, v, z


MAX_VERTICAL_DISPARITY_PX = 0.5


def max_vertical_disparity_px(view: PinholeView, baseline_m: float,
                              min_depth_m: float) -> float:
    """이 뷰를 360 스테레오 쌍으로 자르면 세로 시차가 최대 몇 화소인가.

    C++ equirect_convert.cpp 의 checkRectified() 와 같은 식이다. 유도는 그쪽
    주석에 있고 요약하면: 베이스라인 t = (B,0,0) 을 카메라계로 옮기면
      t_cam = (B cos y, B sin p sin y, B cos p sin y)
    이고 정적 점의 두 영상 사이 세로 이동은
      dv = -(fy/Z) t_y + (v-cy) t_z/Z = (B sin y / Z)(-fy sin p + (v-cy) cos p)
    이다. |v-cy| = H/2 에서 최악이 된다.

    dv 가 **정확히** 0 이 되는 것은 sin(yaw) = 0, 즉 yaw 가 0 또는 180 도일
    때뿐이다. 같은 결론을 축으로도 볼 수 있다: 정렬 스테레오이려면 베이스라인이
    영상 x 축과 평행하고 광축과 직교해야 하는데,
      x_cam = R*(1,0,0) = (cos yaw, 0, -sin yaw)
      z_cam = R*(0,0,1) = (cos pitch*sin yaw, -sin pitch, cos pitch*cos yaw)
    이므로 t . z_cam = B cos(pitch) sin(yaw) = 0 이 조건이고, 그때 t x x_cam 도
    0 이라 평행 조건이 같이 성립한다. pitch 는 베이스라인 축을 도는 회전이라
    조건에 들어오지 않는다 - 위 dv 식에서 pitch 항이 sin(yaw) 에 곱해져만
    나타나는 것과 같은 이야기다.
    """
    s = abs(np.sin(np.deg2rad(view.yaw_deg)))
    sp = abs(np.sin(np.deg2rad(view.pitch_deg)))
    cp = abs(np.cos(np.deg2rad(view.pitch_deg)))
    return float((baseline_m * s / min_depth_m) * (view.fy * sp + (view.height / 2.0) * cp))


def rectifiable_yaw_limit_deg(view: PinholeView, baseline_m: float,
                              min_depth_m: float) -> float:
    """세로 시차가 허용치를 넘기 시작하는 yaw. pitch = 0 에서의 닫힌 해.

    위 식에 pitch = 0 을 넣으면 dv_max = (B sin y / Z)(H/2) 이므로
      sin y_lim = 2 * dv_허용 * Z / (B * H)
    이 값보다 안쪽 yaw 만 스테레오로 쓸 수 있다. 이 한계가 얼마나 좁은지가
    "360 스테레오면 다 되는 것 아닌가" 에 대한 정량적 답이다.
    """
    s = 2.0 * MAX_VERTICAL_DISPARITY_PX * min_depth_m / (baseline_m * view.height)
    if s >= 1.0:
        return 90.0
    return float(np.rad2deg(np.arcsin(s)))
