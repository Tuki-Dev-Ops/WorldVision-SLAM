"""보류집합(held-out) 일반화 검증.

docs/06-results.md 의 상수는 대부분 다섯 개의 9 초 창, 한 데이터셋, 한 센서에서
맞춰졌다 (17 절이 그렇게 적고 있다). 이 스크립트는 그 상수들을 **한 번도 쓰지
않은 시퀀스**에 그대로 대고, 결론이 옮겨가는지만 본다.

원칙 하나: 여기서는 아무것도 튜닝하지 않는다. 상수가 옮겨가지 않으면 그것이
결과다. 숫자가 나빠 보인다고 상수를 고치면 이 실험이 생산하는 유일한 것이
없어진다.

무엇을 먼저 재는가 - 작동 범위 (docs/06-results.md 10.4)
  결론을 말하기 전에 그 시퀀스가 그 결론을 잴 수 있는 상태인지 본다.
  12.2 의 프레임 간격 결함이 정확히 이 자리에서 났다: 데이터가 엔진의 수렴
  반경 밖이면 측정된 것은 엔진이 아니라 데이터다.
    - rgb/depth 짝, 프레임 간격 중앙값
    - 유효 깊이 화소 비율
    - 프레임 간 진리값 운동 vs ECDA 실측 수렴 반경 0.18 m
    - 프레임당 검출 객체 수 vs TCG min_nodes = 4

기준선은 매 표에 남긴다. 그리고 임계는 느슨/엄격을 같이 낸다 - fr1_xyz 는
궤적이 0.698 m 라 상수 추정기도 0.25 m 임계를 76 % 통과한다.

사용:
  python tools/tum_generalize.py envelope <데이터경로> [시퀀스...]
  python tools/tum_generalize.py score <데이터경로> <결과경로> [시퀀스...]
  python tools/tum_generalize.py nees <데이터경로> <결과경로> [시퀀스...]
  python tools/tum_generalize.py scaling <시퀀스경로> <진단디렉토리...>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from wme.eval.trajectory import Trajectory, evaluate_ate, evaluate_rpe  # noqa: E402
from wme.eval.tum import load_sequence, load_trajectory  # noqa: E402
from wme.reference.geometry import SE3  # noqa: E402

# ECDA 실측 수렴 반경 (docs/06-results.md 11.7 / 12.2).
CONVERGENCE_RADIUS = 0.18
# TCG 성좌 성립 최소 노드 수 (ConstellationIndex.hpp).
MIN_NODES = 4


# --- 작동 범위 --------------------------------------------------------------

def envelope(root: Path) -> dict:
    """이 시퀀스가 무엇을 잴 수 있는 상태인지 먼저 본다."""
    seq = load_sequence(root)
    stamps = np.array([f.stamp for f in seq.frames])
    gaps = np.diff(np.sort(stamps))

    gt = seq.trajectory()
    n_gt = len(gt.poses)

    # 진리값이 있는 프레임 사이의 실제 운동. 수렴 반경과 비교할 양이다.
    motion = np.empty(0)
    path_len = 0.0
    if n_gt > 1:
        pos = gt.positions
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        dt = np.diff(gt.stamps)
        # 진리값은 100 Hz 라 그대로 쓰면 프레임 간 운동이 아니다. 30 Hz 로 환산.
        motion = step / np.maximum(dt, 1e-9) * float(np.median(gaps)) if len(gaps) else step
        path_len = float(step.sum())

    # 유효 깊이 비율. 표본 프레임만 읽는다 - 전부 읽을 이유가 없다.
    depth_valid = []
    have_depth = [f for f in seq.frames if f.depth_path is not None]
    for f in have_depth[:: max(1, len(have_depth) // 20)][:20]:
        d = np.asarray(Image.open(f.depth_path))
        depth_valid.append(float(np.count_nonzero(d) / d.size))

    return {
        "name": root.name,
        "frames": len(seq.frames),
        "with_depth": len(have_depth),
        "gt_poses": n_gt,
        "span_s": float(stamps.max() - stamps.min()) if len(stamps) else 0.0,
        "gap_med_ms": float(np.median(gaps) * 1000) if len(gaps) else float("nan"),
        "gap_max_ms": float(gaps.max() * 1000) if len(gaps) else float("nan"),
        "path_len": path_len,
        "motion_med": float(np.median(motion)) if len(motion) else float("nan"),
        "motion_p95": float(np.percentile(motion, 95)) if len(motion) else float("nan"),
        "motion_max": float(motion.max()) if len(motion) else float("nan"),
        "over_radius": float(np.mean(motion > CONVERGENCE_RADIUS)) if len(motion) else float("nan"),
        "depth_valid": float(np.mean(depth_valid)) if depth_valid else float("nan"),
    }


def print_envelope(rows: list[dict]) -> None:
    print(f"{'시퀀스':<44} {'프레임':>6} {'깊이':>6} {'구간s':>6} {'간격ms':>7} "
          f"{'최대ms':>7} {'경로m':>6} {'운동중앙mm':>10} {'운동p95mm':>9} "
          f"{'>18cm%':>7} {'유효깊이%':>9}")
    print("-" * 130)
    for r in rows:
        print(f"{r['name']:<44} {r['frames']:>6} {r['with_depth']:>6} "
              f"{r['span_s']:>6.1f} {r['gap_med_ms']:>7.1f} {r['gap_max_ms']:>7.0f} "
              f"{r['path_len']:>6.2f} {r['motion_med']*1000:>10.1f} "
              f"{r['motion_p95']*1000:>9.1f} {r['over_radius']*100:>7.1f} "
              f"{r['depth_valid']*100:>9.1f}")


# --- 정확도 -----------------------------------------------------------------

def thresholds(errors: np.ndarray) -> tuple[float, float]:
    """느슨(0.25 m) / 엄격(0.10 m) 통과율. 둘 다 내야 궤적 크기에 속지 않는다."""
    return float(np.mean(errors < 0.25)), float(np.mean(errors < 0.10))


def score_one(root: Path, est_path: Path) -> dict | None:
    if not est_path.exists():
        return None
    gt = load_sequence(root).trajectory()
    est = load_trajectory(est_path)
    if len(est.poses) < 3:
        return None
    a = evaluate_ate(est, gt, align=True)
    r = evaluate_rpe(est, gt, delta=1.0)
    loose, tight = thresholds(a.errors)
    return {"ate": a.rmse * 100, "rpe_t": r.trans_rmse * 100,
            "rpe_r": r.rot_rmse_deg, "loose": loose * 100, "tight": tight * 100,
            "n": a.count}


def baseline(root: Path, est_path: Path) -> dict | None:
    """identity - 카메라가 안 움직였다고 가정. 어떤 줄도 이보다 나쁘면 안 된다."""
    if not est_path.exists():
        return None
    gt = load_sequence(root).trajectory()
    est = load_trajectory(est_path)
    ident = Trajectory(est.stamps, [SE3.identity() for _ in est.stamps])
    a = evaluate_ate(ident, gt, align=True)
    loose, tight = thresholds(a.errors)
    return {"ate": a.rmse * 100, "loose": loose * 100, "tight": tight * 100}


def cmd_score(data: Path, results: Path, seqs: list[str], variants: list[str]) -> None:
    print(f"{'시퀀스':<40} {'변형':<10} {'ATE cm':>8} {'RPE cm/s':>9} "
          f"{'RPE °/s':>8} {'<25cm%':>7} {'<10cm%':>7} {'n':>5}")
    print("-" * 100)
    for s in seqs:
        root = data / s
        if not (root / "rgb.txt").exists():
            continue
        shown_base = False
        for v in variants:
            est = results / f"{s}_{v}.txt"
            r = score_one(root, est)
            if r is None:
                continue
            if not shown_base:
                b = baseline(root, est)
                if b:
                    print(f"{s:<40} {'identity':<10} {b['ate']:>8.2f} {'':>9} "
                          f"{'':>8} {b['loose']:>7.1f} {b['tight']:>7.1f}")
                shown_base = True
            print(f"{'':<40} {v:<10} {r['ate']:>8.2f} {r['rpe_t']:>9.2f} "
                  f"{r['rpe_r']:>8.2f} {r['loose']:>7.1f} {r['tight']:>7.1f} "
                  f"{r['n']:>5}")
        print()


# --- 보정 -------------------------------------------------------------------

def cmd_nees(data: Path, results: Path, seqs: list[str], suffix: str) -> None:
    """tum_nees 의 측정 코드를 그대로 부른다. 채점기를 새로 쓰면 비교가 안 된다."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tum_nees import collect, print_one  # noqa: E402

    summary = []
    for s in seqs:
        root = data / s
        diag = results / f"{s}{suffix}_diag.csv"
        if not (root / "rgb.txt").exists() or not diag.exists():
            continue
        frames, meta = collect(root, diag)
        a = print_one(s, frames, meta, True)
        if a:
            summary.append((s, a))

    if summary:
        print("\n\n===== 요약 =====")
        print(f"{'시퀀스':<44} {'n':>4} {'ANEES':>9} {'배율':>8} {'상대오차mm':>10}")
        for name, a in summary:
            print(f"{name:<44} {a['full']['n']:>4} {a['full']['anees']:>9.2f} "
                  f"{a['full']['inflation']:>8.2f} {a['trans_err_med']*1000:>10.2f}")


def cmd_scaling(root: Path, dirs: list[Path], stem: str) -> None:
    """점 개수를 솎아가며 잰 N 스케일링 기울기. 15.1 의 물리적 주장 그 자체다."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tum_nees import collect, scaling_report  # noqa: E402

    runs = []
    for d in dirs:
        p = d / f"{stem}_diag.csv"
        if p.exists():
            runs.append((d.name, collect(root, p)[0]))
    if len(runs) < 3:
        print(f"{stem}: 실행이 3 개 미만이라 기울기를 내지 않는다")
        return
    print(f"\n===== {stem} =====")
    for line in scaling_report(runs):
        print(line)


# --- 검출 밀도 --------------------------------------------------------------

def cmd_detections(token_diags: list[Path]) -> None:
    """프레임당 객체 수 vs min_nodes. 미달이면 TCG 는 통째로 침묵한다."""
    import csv
    print(f"{'시퀀스':<44} {'프레임':>6} {'객체/프레임':>11} {'>=4 비율%':>10} "
          f"{'중앙':>5} {'최대':>5}")
    print("-" * 90)
    for p in token_diags:
        if not p.exists():
            continue
        per_frame: dict[str, int] = {}
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                per_frame[r["stamp"]] = per_frame.get(r["stamp"], 0) + 1
        if not per_frame:
            continue
        c = np.array(list(per_frame.values()))
        print(f"{p.stem:<44} {len(c):>6} {c.mean():>11.2f} "
              f"{np.mean(c >= MIN_NODES)*100:>10.1f} {np.median(c):>5.0f} "
              f"{c.max():>5}")


# --- static_belief 상수 -----------------------------------------------------

# ConfidenceEngine.hpp. sitting/walking 두 시퀀스에서 맞춰진 값이다.
MOTION_NOISE_FLOOR = 0.035


def cmd_static(token_diags: list[Path]) -> None:
    """`motion_noise_floor` 가 옮겨가는가.

    이 상수는 "정지한 물체가 판정 창에서 실제로 보이는 변위" 의 sigma 하한이고,
    그 변위에는 오도메트리 자신의 오차가 공통모드로 섞인다 - 17 절이 순환이라고
    적은 지점이다. 그러니 오도메트리가 더 나쁜 시퀀스에서는 같은 고정 물체가 더
    크게 움직여 보여야 하고, 0.035 로 덮이지 않으면 상수는 옮겨가지 않는 것이다.

    자율 개체(사람)는 제외한다 - 그쪽은 실제로 움직인다.
    """
    import csv
    print(f"{'실행':<52} {'정적토큰':>8} {'창':>6} {'변위중앙mm':>10} "
          f"{'p90mm':>8} {'추정sigma':>9} {'floor초과%':>10}")
    print("-" * 112)
    for p in token_diags:
        if not p.exists():
            continue
        disp, ids = [], set()
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if int(float(r["agent"])) != 0:
                    continue          # 사람은 실제로 움직인다
                d = float(r["su_disp"])
                if float(r["su_updates"]) <= 0 or d <= 0:
                    continue
                disp.append(d)
                ids.add(r["id"])
        if len(disp) < 10:
            continue
        a = np.array(disp)
        # 창 변위는 독립 관측 두 개의 차라 sigma = 중앙값 / sqrt(2) 근사.
        sig = float(np.median(a)) / np.sqrt(2)
        print(f"{p.stem[:52]:<52} {len(ids):>8} {len(a):>6} "
              f"{np.median(a)*1000:>10.1f} {np.percentile(a, 90)*1000:>8.1f} "
              f"{sig*1000:>9.1f} {np.mean(a > MOTION_NOISE_FLOOR)*100:>10.1f}")


def cmd_belief(token_diags: list[Path]) -> None:
    """믿음이 사람과 배경을 실제로 가르는가 (16.7 의 분리비)."""
    import csv
    print(f"{'실행':<52} {'사람믿음':>8} {'배경믿음':>8} {'분리비':>7} "
          f"{'사람>0.4%':>9} {'사람토큰':>8} {'수명중앙':>8}")
    print("-" * 108)
    for p in token_diags:
        if not p.exists():
            continue
        ag, st = [], []
        life: dict[str, int] = {}
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                b = float(r["static_belief"])
                if int(float(r["agent"])) != 0:
                    ag.append(b)
                    life[r["id"]] = life.get(r["id"], 0) + 1
                else:
                    st.append(b)
        if not ag or not st:
            continue
        ma, ms = float(np.mean(ag)), float(np.mean(st))
        print(f"{p.stem[:52]:<52} {ma:>8.3f} {ms:>8.3f} "
              f"{ms/max(ma, 1e-9):>7.2f} {np.mean(np.array(ag) > 0.4)*100:>9.1f} "
              f"{len(life):>8} {np.median(list(life.values())):>8.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("envelope")
    e.add_argument("data")
    e.add_argument("seqs", nargs="*")

    s = sub.add_parser("score")
    s.add_argument("data")
    s.add_argument("results")
    s.add_argument("seqs", nargs="*")
    s.add_argument("--variants", default="maskoff,maskon,masktoken")

    n = sub.add_parser("nees")
    n.add_argument("data")
    n.add_argument("results")
    n.add_argument("seqs", nargs="*")
    n.add_argument("--suffix", default="")

    c = sub.add_parser("scaling")
    c.add_argument("root")
    c.add_argument("dirs", nargs="+")

    d = sub.add_parser("detections")
    d.add_argument("files", nargs="+")

    st = sub.add_parser("static")
    st.add_argument("files", nargs="+")

    bl = sub.add_parser("belief")
    bl.add_argument("files", nargs="+")

    args = ap.parse_args()

    if args.cmd == "envelope":
        data = Path(args.data)
        seqs = args.seqs or sorted(p.name for p in data.iterdir() if (p / "rgb.txt").exists())
        print_envelope([envelope(data / s) for s in seqs])
    elif args.cmd == "score":
        data = Path(args.data)
        seqs = args.seqs or sorted(p.name for p in data.iterdir() if (p / "rgb.txt").exists())
        cmd_score(data, Path(args.results), seqs, args.variants.split(","))
    elif args.cmd == "nees":
        data = Path(args.data)
        seqs = args.seqs or sorted(p.name for p in data.iterdir() if (p / "rgb.txt").exists())
        cmd_nees(data, Path(args.results), seqs, args.suffix)
    elif args.cmd == "scaling":
        root = Path(args.root)
        cmd_scaling(root, [Path(x) for x in args.dirs], root.name)
    elif args.cmd == "detections":
        cmd_detections([Path(x) for x in args.files])
    elif args.cmd == "static":
        cmd_static([Path(x) for x in args.files])
    elif args.cmd == "belief":
        cmd_belief([Path(x) for x in args.files])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
