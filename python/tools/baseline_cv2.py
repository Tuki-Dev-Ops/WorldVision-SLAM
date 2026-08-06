"""A second baseline that nobody in this project wrote.

§22.4 concedes the objection that will always be available against
`wme_tum_baseline`: it is a self-implemented control, so an underperformance can
always be blamed on the implementation rather than the method. Published
settings and an inlier health check are a defence, not a proof.

`cv2.Odometry` (OpenCV 5, `ODOMETRY_TYPE_RGB_DEPTH`) is a **third-party,
published, widely-used dense RGB-D odometry** — the Steinbrücker/Newcombe line
of direct methods. It shares nothing with this repository. If it lands in the
same region as the hand-written ORB baseline, the hand-written one is not
crippled; if it lands somewhere else, that is worth knowing before any claim is
made.

It is also *the same family as WME's Tier 0* — dense, direct, photometric —
which makes it the more informative control of the two: it isolates "this
particular direct method" from "direct methods in general".

Same data path as everything else: same intrinsics, same undistortion, same
depth scale, same rgb/depth association, same keyframe rule.

Usage:
  python tools/baseline_cv2.py [--seq NAME ...] [--out results/bench]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wme.eval.trajectory import Trajectory, evaluate_ate  # noqa: E402
from wme.reference.geometry import SE3  # noqa: E402

from fusion_eval import load_groundtruth  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEPTH_SCALE = 5000.0
KF_DIST = 0.03            # wme_tum_odometry / wme_tum_baseline 와 동일


def intrinsics(name: str):
    if "freiburg2" in name:
        return (520.908620, 521.007327, 325.141442, 249.701764,
                np.array([0.2312, -0.7849, -0.0033, -0.0001, 0.9172]))
    if "freiburg3" in name:
        return (535.4, 539.2, 320.1, 247.6, np.zeros(5))
    return (517.306408, 516.469215, 318.643040, 255.313989,
            np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633]))


def read_index(p: Path):
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        a = line.split()
        out.append((float(a[0]), a[1]))
    return out


def run_sequence(seq: Path, out_txt: Path) -> dict:
    fx, fy, cx, cy, dist = intrinsics(seq.name)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    m1, m2 = cv2.initUndistortRectifyMap(K, dist, None, K, (640, 480), cv2.CV_16SC2)

    rgb = read_index(seq / "rgb.txt")
    dep = read_index(seq / "depth.txt")
    dep_t = np.array([d[0] for d in dep])

    settings = cv2.OdometrySettings()
    settings.setCameraMatrix(K.astype(np.float32))   # CV_32F 를 요구한다
    odo = cv2.Odometry(cv2.ODOMETRY_TYPE_RGB_DEPTH, settings,
                       cv2.OdometryAlgoType_COMMON)

    T_world_cam = SE3.identity()
    T_world_kf = SE3.identity()
    ref = None                       # (gray32, depth32)
    stamps, poses = [], []
    tracked = lost = 0

    for stamp, rel in rgb:
        f = seq / rel
        if not f.exists():
            continue
        j = int(np.argmin(np.abs(dep_t - stamp)))
        if abs(dep_t[j] - stamp) > 0.02:
            continue
        df = seq / dep[j][1]
        if not df.exists():
            continue
        bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
        dpt = cv2.imread(str(df), cv2.IMREAD_UNCHANGED)
        if bgr is None or dpt is None:
            continue

        bgr = cv2.remap(bgr, m1, m2, cv2.INTER_LINEAR)
        dpt = cv2.remap(dpt, m1, m2, cv2.INTER_NEAREST)
        # cv2.Odometry 는 gray 를 CV_8U 로 요구한다. float32 를 넘기면 조용히
        # 실패하는 것이 아니라 예외를 던지지만, 예외를 삼키면 "매 프레임 추적
        # 실패" 가 되어 ATE 가 정확히 do-nothing 바닥값과 같아진다 - 처음 실행이
        # 정확히 그랬다. 바닥값과 같은 수치는 측정이 아니라 신호다.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        z = dpt.astype(np.float32) / DEPTH_SCALE   # 무효 깊이는 0 그대로

        cur = (gray, z)
        T_cur_kf = SE3.identity()
        ok = False
        if ref is not None:
            Rt = np.eye(4)
            try:
                ok, Rt = odo.compute(ref[1], ref[0], cur[1], cur[0], Rt)
            except cv2.error as e:
                # 삼키지 않는다. 형식 오류를 "추적 실패" 로 세면 매 프레임
                # 실패가 되고, 그 궤적은 바닥값과 구분되지 않는다.
                raise SystemExit(f"cv2.Odometry.compute 실패 (입력 형식): {e}")
            if ok and np.all(np.isfinite(Rt)):
                # Rt * src = dst  →  keyframe 좌표계에서 현재 좌표계로.
                T_cur_kf = SE3(np.ascontiguousarray(Rt[:3, :3]),
                               np.ascontiguousarray(Rt[:3, 3]))
        if ok:
            T_world_cam = T_world_kf @ T_cur_kf.inverse()
            tracked += 1
        elif ref is not None:
            lost += 1               # 직전 포즈 유지 - 다른 도구와 같은 규약

        stamps.append(stamp)
        poses.append(T_world_cam)

        if ref is None or np.linalg.norm(T_cur_kf.inverse().t) > KF_DIST or not ok:
            ref = cur
            T_world_kf = T_world_cam

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write("# timestamp tx ty tz qx qy qz qw\n")
        for s, p in zip(stamps, poses):
            q = _quat(p.R)
            fh.write(f"{s:.6f} {p.t[0]:.6f} {p.t[1]:.6f} {p.t[2]:.6f} "
                     f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
    return {"frames": len(poses), "tracked": tracked, "lost": lost,
            "stamps": stamps, "poses": poses}


def _quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0] * 4
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    ap.add_argument("--out", default="results/bench")
    a = ap.parse_args()

    seqs = ([ROOT / "data" / f"rgbd_dataset_{s}" for s in a.seq] if a.seq
            else sorted((ROOT / "data").glob("rgbd_dataset_*")))
    out = ROOT / a.out
    print(f"cv2 {cv2.__version__}  ODOMETRY_TYPE_RGB_DEPTH  (third-party control)")
    print(f"{'sequence':<30}{'ATE':>9}{'frames':>8}{'lost':>7}")
    for seq in seqs:
        if not (seq / "rgb.txt").exists():
            continue
        name = seq.name.replace("rgbd_dataset_", "")
        r = run_sequence(seq, out / f"{name}_cv2rgbd.txt")
        gs, gp = load_groundtruth(seq)
        try:
            ate = evaluate_ate(Trajectory(np.array(r["stamps"]), r["poses"]),
                               Trajectory(gs, gp), align=True).rmse * 100
        except ValueError:
            ate = float("nan")
        print(f"{name:<30}{ate:>8.2f}c{r['frames']:>8}{r['lost']:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
