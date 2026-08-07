"""Run both systems over every TUM sequence and score them into one JSON.

The comparison this produces is odometry-vs-odometry:

  baseline : ORB features -> Hamming descriptor matching -> RANSAC PnP
             (`wme_tum_baseline`) - the classical descriptor pipeline that WME
             declares it does not use, which is what makes it the honest
             control rather than an arbitrary opponent.
  wme      : ECDA Tier-0 direct alignment (`wme_tum_odometry`).

Neither has loop closure or bundle adjustment, both read the same frames
through the same undistortion with the same intrinsics, depth scale, frame
association and keyframe rule. **Neither is ORB-SLAM3** - see the header of
`tools/tum_baseline.cpp` for exactly what is and is not being claimed.

Both are scored here, by a third piece of code that estimated neither, and
against the same "camera never moved" floor the rest of the document uses.

Usage:
  python tools/bench_run.py [--out results/bench] [--max-frames N] [--skip-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wme.eval.trajectory import Trajectory, evaluate_ate  # noqa: E402
from wme.eval.tum import load_trajectory  # noqa: E402
from wme.reference.geometry import SE3, so3_log  # noqa: E402

from fusion_eval import load_groundtruth, interpolate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OPENCV_BIN = Path("C:/opencv-dl/opencv/build/x64/vc16/bin")

SYSTEMS = {
    "baseline": {
        "exe": "wme_tum_baseline.exe",
        "label": "Classical SLAM  /  ORB + PnP",
        "kind": "descriptor",
        "args": [],
    },
    "wme": {
        "exe": "wme_tum_odometry.exe",
        "label": "WorldVision  /  WME Tier 0",
        "kind": "descriptor-free",
        "args": [],
    },
    # 토큰 마스킹까지 켠 구성. 동적 장면이 WME 의 알려진 약점이고(13.1),
    # 그 약점을 메우라고 만든 계층이 이것이므로 빼면 비교가 한쪽으로 기운다.
    "wme_masked": {
        "exe": "wme_tum_odometry.exe",
        "label": "WorldVision  /  + token mask",
        "kind": "descriptor-free",
        # 절대 경로여야 한다. 하위 프로세스의 cwd 는 이 스크립트의 위치이지
        # 저장소 루트가 아니다.
        "args": ["--yolo", str(ROOT / "models" / "yolo11n.onnx"),
                 "--mask-mode", "token"],
    },
}


# 데이터셋마다 필요한 인자가 다르다. 이것을 코드에 박지 않고 여기 모아 두는
# 이유는 25.20 이 기록한 사고 때문이다 - `z > 8.0` 이라는 TUM 전용 상수가 두
# 도구에 박혀 있어서, KITTI 에서 기준선이 프레임당 대응 35 개로 굶고 400 중
# 371 프레임을 놓쳤다. 그대로 발표했으면 WME 의 78 배 승리로 읽혔을 것이다.
#
#   kf-dist    TUM 0.03 m 는 손에 든 카메라 기준이다. 차량은 프레임당 1.5 m 를
#              움직이므로 그 값이면 매 프레임 키프레임이 바뀐다.
#   depth-max  KITTI 스테레오 깊이의 실제 상한은 ~48 m.
#   depth-sigma-rel  sigma_Z = c*Z^2 의 c. 스테레오는 c = sigma_d/(f*B) 로
#              **유도**된다. KITTI gray: 0.3 px / (718.856 * 0.537) = 7.8e-4.
#              튜닝값이 아니라는 점이 중요하다 (25.21).
DATASET_ARGS = {
    "kitti": {
        "common": ["--kf-dist", "1.0", "--depth-max", "60"],
        "wme_only": ["--depth-sigma-rel", "7.8e-4"],
        "skip_systems": ("wme_masked",),   # YOLO 는 TUM 쪽에서만 비교되어 있다
    },
}


def dataset_of(seq: Path) -> str:
    return "kitti" if seq.name.startswith("kitti_") else "tum"


def extra_args(seq: Path, system_key: str) -> list[str]:
    spec = DATASET_ARGS.get(dataset_of(seq))
    if not spec:
        return []
    out = list(spec.get("common", []))
    if system_key.startswith("wme"):
        out += spec.get("wme_only", [])
    return out


def find_build() -> Path:
    for c in ("build/win/tools", "build/msvc/tools"):
        p = ROOT / c
        if (p / "wme_tum_baseline.exe").exists():
            return p
    raise SystemExit("wme_tum_baseline.exe 를 찾지 못했다 - 먼저 빌드해야 한다")


def run_system(exe: Path, seq: Path, out_traj: Path, diag: Path,
               max_frames: int, extra: list[str] | None = None) -> dict:
    env = dict(os.environ)
    if OPENCV_BIN.exists():
        env["PATH"] = str(OPENCV_BIN) + os.pathsep + env.get("PATH", "")
    cmd = [str(exe), str(seq), str(out_traj), "--diag", str(diag)]
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]
    cmd += extra or []
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       encoding="utf-8", errors="replace")
    wall = time.time() - t0
    if p.returncode != 0 or not out_traj.exists():
        return {"ok": False, "stderr": (p.stderr or "")[-400:], "wall_s": wall}
    return {"ok": True, "wall_s": wall, "stdout": (p.stdout or "")[-400:]}


def rel_errors(est: Trajectory, gt_stamps, gt_poses) -> dict:
    """Per-frame relative-pose error against ground truth.

    Frame-to-frame rather than against a keyframe, so the two systems are
    compared on a quantity neither of them chose.

    보간 허용 간격은 시퀀스 자신의 표본 간격에서 정한다. `interpolate` 의
    기본 50 ms 는 30 Hz TUM 기준이고, KITTI 를 stride 2 로 쓰면 표본 간격이
    200 ms 라 **모든** 보간이 거부된다 - RPE 열이 통째로 비고, 뷰어에는 실패가
    아니라 빈칸으로 나타난다. 25.20 의 깊이 상한과 같은 종류의 상수다.
    """
    step = float(np.median(np.diff(gt_stamps))) if len(gt_stamps) > 1 else 0.0
    gap = max(0.05, 3.0 * step)
    trans, rot, stamps = [], [], []
    for i in range(1, len(est.poses)):
        g1 = interpolate(gt_stamps, gt_poses, est.stamps[i - 1], max_gap=gap)
        g2 = interpolate(gt_stamps, gt_poses, est.stamps[i], max_gap=gap)
        if g1 is None or g2 is None:
            continue
        gt_rel = g1.inverse() @ g2
        es_rel = est.poses[i - 1].inverse() @ est.poses[i]
        d = es_rel.inverse() @ gt_rel
        if not np.all(np.isfinite(d.t)):
            continue
        trans.append(float(np.linalg.norm(d.t)))
        rot.append(float(np.linalg.norm(so3_log(d.R))))
        stamps.append(float(est.stamps[i]))
    return {"stamps": stamps, "trans": trans, "rot": rot}


def read_diag(path: Path) -> list[dict]:
    if not path.exists():
        return []
    import csv
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [dict(r) for r in csv.DictReader(f)]


def run_status(est: Trajectory, gt: Trajectory | None = None) -> str:
    """ATE 를 숫자로 보고해도 되는 실행인가.

    두 경우는 측정이 아니라 신호이고, 숫자로 적으면 둘 다 그럴듯해 보인다:

    - **no_output**: 궤적이 전부 항등. 엔진이 아무 것도 내지 않았다는 뜻인데,
      "카메라가 안 움직였다" 바닥값과 **정확히 같은 점수**를 받으므로 표에서는
      선전하는 것처럼 보인다. 실제로 `nostructure_notexture_far` 에서 WME 가
      바닥값과 소수점까지 같은 값을 냈고, 그것이 이 검사를 만든 이유다.
    - **diverged**: 정렬 후 ATE 는 유한한 값으로 나오지만 그 수치는 아무 것도
      뜻하지 않는다.

    발산 판정의 기준은 **진리값 자신** 이다. 예전에는 1 km 짜리 상수였는데,
    그 숫자는 "실내 45 초 시퀀스가 1 km 를 갈 수는 없다" 는 한 데이터셋에 대한
    직관이고 KITTI 같은 실외 주행에는 아무 의미가 없다. 게다가 너무 헐거웠다 -
    `sitting_halfsphere` 에서 마스크 변형이 **48.9 m** 를 돌아다녔는데, 카메라가
    팔 길이 반경으로 도는 시퀀스에서 그것도 불가능하다. 1 km 아래라는 이유로
    보통 숫자로 실렸다 (25.22).

    경로 길이 비는 시퀀스마다 스스로 스케일을 갖는다. 실제 오도메트리 드리프트는
    경로를 늘리는 것이 아니라 굽히는 것이고, 길이를 유지한 채 틀린 방향으로 가는
    실패는 ATE 가 잡는다.

    문턱 6 배는 추측이 아니라 **측정된 빈 구간에서** 골랐다. 20 시퀀스 x 3 시스템
    56 개 실행의 비 분포는 두 덩어리로 갈라진다:

        건강한 쪽  0.62 ... 3.67   (최대: nostructure_texture_far / ORB)
        --------- 빈 구간 ---------
        실패한 쪽  10.70, 153.03, 7.3e11, 5.0e13

    6 은 그 사이에 있고 로그 척도로 양쪽에서 비슷하게 떨어져 있다 - 최악의 건강한
    실행보다 1.6 배 위, 가장 가까운 실패보다 1.8 배 아래. 10 을 쓰면 10.70 과 7 %
    차이가 되는데, 이 저장소가 반복해 적은 대로 **경계에서 재면 재는 것이 현상이
    아니라 반올림이다** (10.4).
    """
    P = est.positions
    if len(P) < 2:
        return "no_output"
    path = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    if path <= 1e-9:
        return "no_output"
    if not np.all(np.isfinite(P)):
        return "diverged"
    if gt is not None and len(gt.positions) >= 2:
        gt_path = float(np.linalg.norm(np.diff(gt.positions, axis=0), axis=1).sum())
        if gt_path > 1e-6 and path > 6.0 * gt_path:
            return "diverged"
    elif float(np.abs(P).max()) > 1e3:
        # 진리값이 없을 때만 옛 상수로 물러난다.
        return "diverged"
    return "ok"


def ate_series(est: Trajectory, gt: Trajectory) -> tuple[float, list[float]]:
    r = evaluate_ate(est, gt, align=True)
    errs = getattr(r, "errors", None)
    return float(r.rmse), ([float(x) for x in errs] if errs is not None else [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bench")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--allow-partial", action="store_true",
                    help="인덱스보다 프레임이 적은 시퀀스도 실행한다 "
                         "(결과는 '전체 시퀀스' 가 아니다)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="시퀀스 이름 또는 데이터셋(tum/kitti) 으로 거른다")
    # --only 를 주면 합치는 것이 기본이다. 옛 기본값(덮어쓰기)은 "한 시퀀스만
    # 확인" 이 나머지 열아홉 개의 결과를 지우는 동작이었고, 실제로 그렇게
    # benchmark.json 하나를 날렸다. 25.22 가 다루는 실패 양식 그대로다:
    # 조용한 축소는 "그 시퀀스에서 아무 일도 없었다" 로 읽힌다.
    ap.add_argument("--merge", action=argparse.BooleanOptionalAction, default=None,
                    help="기존 benchmark.json 의 다른 시퀀스를 보존한 채 합친다 "
                         "(--only 를 주면 기본 켜짐, --no-merge 로 끈다)")
    ap.add_argument("--skip-run", action="store_true",
                    help="재추정 없이 기존 궤적만 다시 채점한다")
    args = ap.parse_args()
    if args.merge is None:
        args.merge = args.only is not None

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    build = find_build()

    seqs = sorted(
        (p for pat in ("rgbd_dataset_*", "kitti_*")
         for p in (ROOT / "data").glob(pat)
         if (p / "rgb.txt").exists() and (p / "groundtruth.txt").exists()),
        key=lambda p: (dataset_of(p), p.name))
    if args.only:
        want = set(args.only)
        seqs = [p for p in seqs
                if p.name in want or p.name.replace("rgbd_dataset_", "") in want
                or dataset_of(p) in want]
    if not seqs:
        raise SystemExit("data/ 에 실행할 시퀀스가 없다")

    # 인덱스가 약속한 프레임이 실제로 디스크에 있는지 **여기서** 본다.
    #
    # 없으면 imread 가 빈 Mat 를 주고 도구는 그 프레임을 조용히 건너뛴다.
    # 궤적은 짧아지지만 ATE 는 계속 계산되고 표에 실린다 - 06-results.md
    # 25.22 는 그렇게 TUM 16 개 중 13 개가 평균 35 % 만 채점된 채로 스무 개
    # 절을 통과한 기록이다. 가장 심한 것은 4042 프레임 중 258 개였다.
    #
    # check_datasets.py 는 그 전에도 있었지만 아무도 부르지 않았다. 검사는
    # 실행 경로 위에 있어야 검사다.
    incomplete = []
    for seq in (() if args.allow_partial else seqs):
        n_idx = n_have = 0
        for kind in ("rgb", "depth"):
            idx = [ln.split()[1] for ln in
                   (seq / f"{kind}.txt").read_text(encoding="utf-8",
                                                   errors="replace").splitlines()
                   if ln.strip() and not ln.startswith("#") and len(ln.split()) >= 2]
            n_idx += len(idx)
            n_have += sum(1 for rel in idx if (seq / rel).exists())
        if n_idx and n_have < n_idx:
            incomplete.append((seq.name, n_have, n_idx))
    if incomplete:
        for name, have, idx in incomplete:
            print(f"  {name}: {have}/{idx} 프레임 ({100.0*have/idx:.1f} %)",
                  file=sys.stderr)
        raise SystemExit(
            f"{len(incomplete)} 개 시퀀스가 인덱스보다 적은 프레임을 갖고 있다. "
            "이대로 재면 '전체 시퀀스' 가 아닌 결과가 그렇게 보이는 표로 나온다 - "
            "python/tools/tum_fetch.py 로 다시 받거나, 의도한 것이면 --allow-partial "
            "을 준다.")

    # 재채점(--skip-run)에는 벽시계가 없다. 이전 실행의 측정값을 이어받지 않으면
    # 타이밍이 통째로 사라지고, 그 자리를 0 으로 채우면 "무한히 빠르다" 가 된다.
    prev: dict[str, dict] = {}
    prev_p = out_dir / "benchmark.json"
    if args.skip_run and prev_p.exists():
        try:
            old = json.loads(prev_p.read_text(encoding="utf-8"))
            prev = {s["name"]: s.get("runs", {}) for s in old.get("sequences", [])}
        except Exception:
            prev = {}

    report = {"sequences": [], "systems": SYSTEMS}
    # --only 로 일부만 돌릴 때 나머지를 지우지 않는다. 지우면 뷰어가 조용히
    # 축소되고, 그것은 "그 시퀀스에서 아무 일도 없었다" 로 읽힌다.
    keep: list[dict] = []
    if args.merge and prev_p.exists():
        try:
            old = json.loads(prev_p.read_text(encoding="utf-8"))
            ran = {s.name.replace("rgbd_dataset_", "") for s in seqs}
            keep = [e for e in old.get("sequences", []) if e["name"] not in ran]
        except Exception:
            keep = []

    for seq in seqs:
        name = seq.name.replace("rgbd_dataset_", "")
        gt_stamps, gt_poses = load_groundtruth(seq)
        gt = Trajectory(gt_stamps, gt_poses)
        entry = {"name": name, "runs": {}}
        print(f"\n=== {name} ===")

        for key, meta in SYSTEMS.items():
            traj_p = out_dir / f"{name}_{key}.txt"
            diag_p = out_dir / f"{name}_{key}_diag.csv"
            if args.skip_run and traj_p.exists():
                run = {"ok": True, "wall_s": None}
            else:
                if key in DATASET_ARGS.get(dataset_of(seq), {}).get("skip_systems", ()):
                    entry["runs"][key] = {"ok": False, "skipped": True}
                    print(f"  {key:9} 건너뜀 ({dataset_of(seq)})")
                    continue
                run = run_system(build / meta["exe"], seq, traj_p, diag_p,
                                 args.max_frames,
                                 list(meta.get("args") or []) + extra_args(seq, key))
            if not run["ok"]:
                print(f"  {key:9} 실패: {run.get('stderr','')[:120]}")
                entry["runs"][key] = {"ok": False}
                continue

            est = load_trajectory(traj_p)
            ate, ate_errs = ate_series(est, gt)
            rel = rel_errors(est, gt_stamps, gt_poses)
            diag = read_diag(diag_p)

            def col(k: str) -> list[float]:
                out = []
                for r in diag:
                    try:
                        out.append(float(r[k]))
                    except (KeyError, ValueError):
                        pass
                return out

            ms = col("ms")
            status = run_status(est, gt)
            entry["runs"][key] = {
                "ok": True,
                "status": status,
                "ate_rmse_cm": ate * 100.0,
                "ate_errors_cm": [e * 100.0 for e in ate_errs],
                "rpe_trans_median_mm": float(np.median(rel["trans"])) * 1000.0
                if rel["trans"] else None,
                "rpe_trans_p95_mm": float(np.percentile(rel["trans"], 95)) * 1000.0
                if rel["trans"] else None,
                "rpe_rot_median_deg": float(np.degrees(np.median(rel["rot"])))
                if rel["rot"] else None,
                "frames": len(est.poses),
                "stamps": [float(s) for s in est.stamps],
                "traj": [[float(p.t[0]), float(p.t[1]), float(p.t[2])]
                         for p in est.poses],
                "rel": rel,
                # 두 시스템의 진단 열이 달라 diag 의 ms 는 서로 다른 구간을
                # 잰다(한쪽은 I/O 제외). 헤드라인 수치는 양쪽 모두에 대해
                # 같은 방식으로 재는 벽시계/프레임을 쓴다.
                "ms_per_frame": (run["wall_s"] / len(est.poses) * 1000.0)
                if run.get("wall_s") and len(est.poses)
                else (prev.get(name, {}).get(key, {}) or {}).get("ms_per_frame"),
                "ms_median_internal": float(np.median(ms)) if ms else None,
                "wall_s": run.get("wall_s"),
                # 시스템마다 진단 열이 다르므로 있는 것만 담는다.
                "diag_extra": {k: (float(np.median(col(k))) if col(k) else None)
                               for k in ("keypoints", "matches", "pnp_inliers",
                                         "inlier_ratio", "reproj_rmse",
                                         "points", "inliers", "rmse",
                                         "observable_dof", "cond")},
                "track_lost": int(sum(1 for r in diag
                                      if r.get("track_ok") == "0")) if diag else None,
            }
            e = entry["runs"][key]
            tag = "" if status == "ok" else f"   <-- {status.upper()} (ATE is meaningless)"
            print(f"  {key:11} ATE {ate*100:7.2f} cm   "
                  f"RPE {e['rpe_trans_median_mm'] or float('nan'):6.2f} mm   "
                  f"{e['ms_per_frame'] or float('nan'):6.1f} ms/frame{tag}")

        # do-nothing 바닥값. 시퀀스마다 다시 계산한다 - 하나로 고정하면
        # 움직임이 적은 시퀀스에서 비교가 조용히 무의미해진다.
        any_run = next((r for r in entry["runs"].values() if r.get("ok")), None)
        if any_run:
            stamps = np.array(any_run["stamps"])
            ident = Trajectory(stamps, [SE3.identity() for _ in stamps])
            entry["identity_ate_cm"] = float(evaluate_ate(ident, gt, align=True).rmse) * 100.0
            gtp = []
            for s in stamps:
                g = interpolate(gt_stamps, gt_poses, float(s))
                gtp.append([float(g.t[0]), float(g.t[1]), float(g.t[2])] if g else None)
            entry["gt_traj"] = gtp
            print(f"  {'identity':9} ATE {entry['identity_ate_cm']:7.2f} cm  (do-nothing floor)")
        entry["dataset"] = dataset_of(seq)
        report["sequences"].append(entry)

    if keep:
        for e in keep:
            e.setdefault("dataset", "tum")
        report["sequences"] = keep + report["sequences"]
        report["sequences"].sort(key=lambda e: (e.get("dataset", "tum"), e["name"]))
        print(f"\n기존 {len(keep)} 시퀀스를 보존해 합쳤다")

    p = out_dir / "benchmark.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    print(f"\n저장: {p}  ({p.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
