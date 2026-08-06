"""TCG(Tier 1) 실데이터 재지역화 채점기.

추정은 C++ (wme_tum_relocalize), 채점은 여기. 같은 코드로 둘 다 하면 두 쪽의
버그가 서로를 가려 준다.

이 스크립트가 먼저 답해야 하는 질문은 "재현율이 얼마냐" 가 아니라
**"측정이 일어나기는 했느냐"** 다. TUM 실내 장면에 YOLO 검출 객체가 min_nodes
미만이면 질의는 통째로 침묵하고, 그 침묵은 "후보 없음 = 보수적 기각" 처럼
보이는 그럴듯한 숫자로 위장된다 (docs/06-results.md 10.4).
그래서 출력 순서를 고정한다.

  1. 프레임당 객체 수 분포와 min_nodes 충족률   <- 작동 범위 자체
  2. 자기질의 결과                              <- 측정기가 살아 있는가
  3. 재현율 / 정밀도 / 오탐률                    <- 1,2 가 통과했을 때만 의미

게이트 열의 의미:
  none        아무 판정 없이 1위 후보를 그대로 받는다 (상한)
  score-ratio 예전 C++ 규칙(score2 > 0.85*score1 이면 기각). 조밀 지도에서
              정대응을 전멸시키는 것을 보이기 위해 남겨 둔다
  pose-agree  1·2위 포즈 거리로 자르는 파이썬 근사 (참고용)
  cpp         **C++ ConstellationIndex::query() 가 실제로 내린 결정.**
              이것이 성적표이고 나머지는 그 결정의 분해다

임계는 느슨한 값과 엄격한 값을 항상 같이 낸다. fr1_xyz 처럼 궤적이 0.7 m 인
시퀀스에서는 0.25 m 임계를 상수 추정기도 76 % 통과하므로, 느슨한 임계의
재현율만 보면 알고리즘이 아니라 궤적 크기를 읽게 된다.

사용:
  python tools/tcg_eval.py <결과접두사> [<결과접두사> ...]
    [--trans-thresh 0.25] [--rot-thresh 15] [--tight-trans 0.10]
    [--tight-rot 10] [--min-nodes 4] [--min-conf 0.0]
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np


# --- 입출력 ----------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_widths(rows: list[dict], name: str) -> bool:
    """행의 칸 수가 헤더와 맞는지 확인한다.

    CSV 를 손으로 쓰면 쉼표 하나가 빠지고, 그러면 뒤의 진리값 열이 통째로
    한 칸씩 밀린다. 밀린 값도 숫자로 파싱되므로 채점기는 조용히 다른 열을
    읽고 그럴듯한 오답을 낸다 (docs/06-results.md 10.4). 여기서 먼저 막는다.
    """
    bad = sum(1 for r in rows if None in r or any(v is None for v in r.values()))
    if bad:
        print(f"  ** {name}: 헤더와 칸 수가 다른 행 {bad}/{len(rows)} 개. "
              f"이 파일로 낸 수치는 측정이 아니다.")
    return bad == 0


def fnum(row: dict, key: str, default: float = float("nan")) -> float:
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def inum(row: dict, key: str, default: int = 0) -> int:
    v = fnum(row, key, float(default))
    return default if math.isnan(v) else int(v)


# --- 포즈 오차 --------------------------------------------------------------

def quat_angle_deg(qa: np.ndarray, qb: np.ndarray) -> float:
    """두 단위 쿼터니언 사이의 회전각(도). 부호 모호성을 절대값으로 흡수한다."""
    d = abs(float(np.dot(qa, qb)))
    return 2.0 * math.degrees(math.acos(min(1.0, max(-1.0, d))))


def pose_of(row: dict, prefix: str) -> tuple[np.ndarray, np.ndarray] | None:
    t = [fnum(row, f"{prefix}_t{a}") for a in "xyz"]
    q = [fnum(row, f"{prefix}_q{a}") for a in "xyzw"]
    if any(math.isnan(v) for v in t + q):
        return None
    qn = np.array(q, dtype=float)
    n = float(np.linalg.norm(qn))
    if n < 1e-9:
        return None
    return np.array(t, dtype=float), qn / n


# --- 1장: 작동 범위 ---------------------------------------------------------

def report_envelope(frames: list[dict], min_nodes: int) -> dict:
    """프레임당 검출/노드 수 분포. 평균 하나로는 침묵을 볼 수 없으므로 분포를 낸다."""
    pos = [r for r in frames if r.get("phase") in ("map", "query")]
    if not pos:
        return {}

    det = np.array([inum(r, "n_det") for r in pos])
    nod = np.array([inum(r, "n_frame_nodes") for r in pos])      # 깊이까지 있는 3D 노드
    cov = np.array([inum(r, "n_covis_stable") for r in pos])     # 토큰 경로(아키텍처 의도)
    stable = np.array([inum(r, "n_stable") for r in pos])

    print(f"  프레임 {len(pos)} 개")
    print(f"  {'':22s} {'평균':>6s} {'중앙':>5s} {'최소':>5s} {'최대':>5s}   분포(개수:프레임수)")
    for name, arr in (("YOLO 검출", det), ("3D 노드(깊이有)", nod),
                      ("공시야 안정토큰", cov)):
        hist = Counter(int(v) for v in arr)
        tail = " ".join(f"{k}:{hist[k]}" for k in sorted(hist)[:14])
        print(f"  {name:22s} {arr.mean():6.2f} {np.median(arr):5.1f} "
              f"{arr.min():5d} {arr.max():5d}   {tail}")

    ok_nodes = float((nod >= min_nodes).mean())
    ok_covis = float((cov >= min_nodes).mean())
    print(f"  min_nodes={min_nodes} 충족률:  3D 노드 {ok_nodes*100:5.1f} %"
          f"   공시야 안정토큰 {ok_covis*100:5.1f} %")
    if stable.max() == 0:
        print("  ** 경고: isStableLandmark() 를 통과한 토큰이 한 프레임도 없다.")
        print("     ConstellationIndex::buildFrom 은 언제나 빈 성좌를 낸다 -")
        print("     이 경로로 잰 값은 알고리즘이 아니라 침묵의 측정이다.")

    # 클래스 구성. 성좌의 변별력은 클래스 조합에서 나오므로 무엇이 잡히는지가 중요하다.
    cc: Counter = Counter()
    for r in pos:
        for c in (r.get("classes") or "").split(";"):
            if c:
                cc[c] += 1
    total = max(1, len(pos))
    top = ", ".join(f"{k} {v/total:.2f}" for k, v in cc.most_common(10))
    print(f"  프레임당 클래스 출현빈도(상위10): {top}")
    return {"nodes_ok": ok_nodes, "covis_ok": ok_covis, "det_mean": float(det.mean())}


# --- 2장: 측정기 자체 검사 ---------------------------------------------------

def report_selftest(rows: list[dict]) -> bool:
    """등록된 Place 를 자기 노드로 질의하면 자기 자신이 나와야 한다.

    여기서 실패하면 아래 재현율은 알고리즘에 대한 진술이 아니다.
    """
    if not rows:
        print("  자기질의 기록 없음")
        return False
    hit = sum(1 for r in rows if inum(r, "top_place") == inum(r, "place_id"))
    te = np.array([fnum(r, "trans_err", float("nan")) for r in rows])
    te = te[~np.isnan(te)]
    acc = sum(1 for r in rows if inum(r, "accept") == 1)
    acc_self = sum(1 for r in rows
                   if inum(r, "accept") == 1 and inum(r, "acc_place") == inum(r, "place_id"))
    print(f"  자기 장소 복원 {hit}/{len(rows)}"
          f"   복원 변환 |t| 최대 {te.max() if te.size else float('nan'):.4f} m")
    # 채택 규칙도 같이 통과해야 한다. 조밀 지도에서 이웃 장소를 "모호"로 읽는
    # 규칙은 자기질의부터 죽이고, 그러면 아래 재현율은 규칙이 아니라 침묵이다.
    print(f"  채택 규칙 통과 {acc}/{len(rows)}  (그 중 자기 장소 {acc_self})")
    if hit != len(rows):
        print("  ** 자기질의가 완전하지 않다. 아래 수치는 해석 불가.")
    return hit == len(rows)


# --- 3장: 재지역화 성적 -----------------------------------------------------

GATES = ("none", "score-ratio", "pose-agree", "cpp")

# 게이트별로 어느 포즈를 채점할지. cpp 게이트는 C++ 이 실제로 내놓은 포즈를 쓴다.
GATE_POSE = {"none": "est", "score-ratio": "est", "pose-agree": "est", "cpp": "acc"}


def gate_accept(row: dict, gate: str, ratio: float, agree_m: float) -> bool:
    """상위 후보를 받아들일지 결정한다. 게이트별로 따로 잰다.

    score-ratio 는 예전 ConstellationIndex::query() 에 하드코딩되어 있던 규칙이다.
    pose-agree 는 그 대안을 파이썬에서 흉내 낸 것 - 참고용으로만 남긴다.
    cpp 는 C++ 이 실제로 내린 결정(accept 열)이다. **이것이 진짜 성적표다**;
    나머지는 그 결정이 어디서 왔는지 보기 위한 분해다.
    """
    if gate == "cpp":
        return inum(row, "accept") == 1
    if inum(row, "n_cand") == 0:
        return False
    if gate == "none":
        return True
    s1, s2 = fnum(row, "score", 0.0), fnum(row, "score2", 0.0)
    if gate == "score-ratio":
        return not (s2 > s1 * ratio)
    if gate == "pose-agree":
        if inum(row, "place2_id") == 0:
            return True
        p1, p2 = pose_of(row, "est"), pose_of(row, "est2")
        if p1 is None or p2 is None:
            return True
        return float(np.linalg.norm(p1[0] - p2[0])) <= agree_m
    return False


def query_nodes(row: dict) -> int:
    """실제로 질의에 들어간 성좌 크기.

    다중 프레임 질의에서는 창을 합친 n_win 이 진짜 질의 크기다. n_nodes(단일
    프레임)로 세면 오탐률의 분모가 창 크기에 따라 흔들려 w1/w5 를 비교할 수 없다.
    """
    w = inum(row, "n_win", 0)
    return w if w > 0 else inum(row, "n_nodes", 0)


def report_reloc(queries: list[dict], gate: str, trans_thresh: float,
                 rot_thresh: float, min_nodes: int, ratio: float,
                 agree_m: float, min_conf: float = 0.0) -> dict:
    pos = [r for r in queries if r.get("tag") == "pos"]
    neg = [r for r in queries if r.get("tag") == "neg"]
    pfx = GATE_POSE[gate]

    def passes(r: dict) -> bool:
        if not gate_accept(r, gate, ratio, agree_m):
            return False
        # 신뢰도 하한은 게이트와 직교한 축이다. 0 이면 아무것도 자르지 않는다.
        key = "conf" if gate == "cpp" else "score"
        return fnum(r, key, 0.0) >= min_conf

    n_q = len(pos)
    silent_few = sum(1 for r in pos if query_nodes(r) < min_nodes)
    accepted, correct, errs, rots = 0, 0, [], []
    for r in pos:
        if not passes(r):
            continue
        est, gt = pose_of(r, pfx), pose_of(r, "gt")
        if est is None or gt is None:
            continue
        accepted += 1
        te = float(np.linalg.norm(est[0] - gt[0]))
        re = quat_angle_deg(est[1], gt[1])
        errs.append(te)
        rots.append(re)
        if te <= trans_thresh and re <= rot_thresh:
            correct += 1

    n_neg = len(neg)
    neg_acc = sum(1 for r in neg if passes(r))
    neg_nodes_ok = sum(1 for r in neg if query_nodes(r) >= min_nodes)

    return {
        "gate": gate,
        "n_query": n_q,
        "silent_few_nodes": silent_few,
        "accepted": accepted,
        "correct": correct,
        "recall": correct / n_q if n_q else float("nan"),
        "precision": correct / accepted if accepted else float("nan"),
        "fp_inmap": (accepted - correct) / n_q if n_q else float("nan"),
        "med_trans": float(np.median(errs)) if errs else float("nan"),
        "med_rot": float(np.median(rots)) if rots else float("nan"),
        "n_neg": n_neg,
        "neg_nodes_ok": neg_nodes_ok,
        "neg_accepted": neg_acc,
        "fp_outmap": neg_acc / neg_nodes_ok if neg_nodes_ok else float("nan"),
    }


def report_baseline(queries: list[dict], places: list[dict], trans_thresh: float,
                    rot_thresh: float, quiet: bool = False) -> float | None:
    """측정이 변별하는지 확인하는 대조군.

    fr1_xyz 처럼 카메라가 한 뼘 안에서만 움직이는 시퀀스에서는 "지도 장소들의
    평균 포즈" 를 늘 답하는 상수 추정기도 임계 안에 들어온다. 그러면 재현율은
    알고리즘이 아니라 임계와 궤적 크기의 함수다. 상수 기준선을 같이 내지 않으면
    그 사실을 볼 수 없다 (docs/06-results.md 10.4).
    """
    pos = [r for r in queries if r.get("tag") == "pos"]
    gts = [pose_of(r, "gt") for r in pos]
    gts = [g for g in gts if g is not None]
    if not gts or not places:
        return None

    anchors = np.array([[fnum(p, "tx"), fnum(p, "ty"), fnum(p, "tz")] for p in places])
    aq = np.array([[fnum(p, "qx"), fnum(p, "qy"), fnum(p, "qz"), fnum(p, "qw")]
                   for p in places])
    gt_t = np.array([g[0] for g in gts])

    span_gt = float(np.linalg.norm(gt_t.max(axis=0) - gt_t.min(axis=0)))
    span_map = float(np.linalg.norm(anchors.max(axis=0) - anchors.min(axis=0)))

    # 상수 기준선: 지도 앵커의 평균 위치 + 평균에 가장 가까운 앵커의 자세
    c_t = anchors.mean(axis=0)
    c_q = aq[int(np.argmin(np.linalg.norm(anchors - c_t, axis=1)))]
    c_q = c_q / max(1e-9, float(np.linalg.norm(c_q)))
    b_te = np.array([float(np.linalg.norm(g[0] - c_t)) for g in gts])
    b_re = np.array([quat_angle_deg(g[1], c_q) for g in gts])
    b_ok = float(((b_te <= trans_thresh) & (b_re <= rot_thresh)).mean())
    if quiet:
        return b_ok

    print(f"  대조군: 진리 궤적 대각 {span_gt:.3f} m, 지도 앵커 대각 {span_map:.3f} m")
    print(f"  대조군: '항상 지도 중심 포즈' 상수 추정기 -> 중앙오차 "
          f"{np.median(b_te):.3f} m / {np.median(b_re):.2f}°,  임계 통과율 {b_ok*100:.1f} %")
    if b_ok > 0.5:
        print("  ** 상수 추정기가 절반 넘게 통과한다. 이 임계에서 재현율은 "
              "알고리즘이 아니라 궤적 크기를 재고 있다.")

    # 장소 선택이 실제로 변별하는가. 아무 장소나 골라도 포즈 오차가 작을 수 있으므로
    # "고른 장소" 와 "최선의 장소" 와 "무작위 장소" 를 나란히 본다.
    by_id = {inum(p, "place_id"): np.array([fnum(p, "tx"), fnum(p, "ty"), fnum(p, "tz")])
             for p in places}
    sel, best, rnd = [], [], []
    for r, g in zip([r for r in pos if pose_of(r, "gt") is not None], gts):
        pid = inum(r, "place_id")
        if pid == 0 or pid not in by_id:
            continue
        d = np.linalg.norm(anchors - g[0], axis=1)
        sel.append(float(np.linalg.norm(by_id[pid] - g[0])))
        best.append(float(d.min()))
        rnd.append(float(d.mean()))
    if sel:
        print(f"  장소 선택: 고른 장소까지 {np.median(sel):.3f} m | "
              f"최선 {np.median(best):.3f} m | 무작위 평균 {np.median(rnd):.3f} m")
    return b_ok


# 시험할 신뢰도 신호. (열 이름, 표시명, 부호)
# 부호 +1 은 "클수록 좋다", -1 은 "클수록 나쁘다". 부호를 맞춰야 여러 신호의
# corr 을 같은 방향으로 읽을 수 있다.
CONF_SIGNALS = (
    ("score",     "score",     +1),
    ("rms",       "rms",       -1),
    ("explained", "explained", +1),
    ("chi2",      "chi2/dof",  -1),
    ("agree",     "agree",     +1),
    ("support",   "support",   +1),
    ("rival",     "rival",     -1),
    ("margin",    "margin",    +1),
    ("n_inl",     "inliers",   +1),
    ("conf",      "confidence", +1),
)


def confidence_rows(queries: list[dict], trans_thresh: float,
                    rot_thresh: float) -> list[dict]:
    """후보가 나온 질의마다 (신호들, 실제 오차, 정답여부) 를 모은다."""
    rows = []
    for r in queries:
        if r.get("tag") != "pos" or inum(r, "n_cand") == 0:
            continue
        est, gt = pose_of(r, "est"), pose_of(r, "gt")
        if est is None or gt is None:
            continue
        te = float(np.linalg.norm(est[0] - gt[0]))
        re = quat_angle_deg(est[1], gt[1])
        d = {k: fnum(r, k, 0.0) for k, _, _ in CONF_SIGNALS}
        d["err"] = te
        d["ok"] = bool(te <= trans_thresh and re <= rot_thresh)
        rows.append(d)
    return rows


def report_confidence(queries: list[dict], trans_thresh: float,
                      rot_thresh: float) -> dict:
    """어떤 신호가 실제 포즈 오차를 예측하는가.

    예측하지 못하면 임계를 어디에 놓아도 정밀도를 살 수 없다 - 그건 튜닝 문제가
    아니라 게이트가 재는 양이 틀렸다는 뜻이다. 카이제곱/rms 는 "대응이 서로
    일관적인가" 를 재지 성좌가 옳은 성좌인지를 재지 않는다. agree/support/margin
    은 다른 후보와의 *교차* 증거라 그 한계 밖에 있다 - 그게 이 표의 요점이다.
    """
    rows = confidence_rows(queries, trans_thresh, rot_thresh)
    if len(rows) < 5:
        print("  신뢰도 예측력: 표본 부족")
        return {}

    er = np.array([r["err"] for r in rows])
    ok = np.array([r["ok"] for r in rows])

    def corr(a, b):
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    out = {}
    print(f"  신뢰도 예측력 (표본 {len(rows)}). corr 은 부호를 맞춰 '클수록 정확' 방향으로 "
          f"보정했다 - 음수일수록 좋은 예측기.")
    line = []
    for key, name, sign in CONF_SIGNALS:
        v = np.array([r[key] for r in rows]) * sign
        c = corr(v, er)
        out[key] = c
        line.append(f"{name}={c:+.2f}")
    print("   corr(신호, 오차): " + "  ".join(line))

    # 정밀도/반환수 트레이드오프. 상관 하나로는 임계를 어디 두어야 하는지 모른다.
    for key, name, sign in (("score", "score", +1), ("conf", "confidence", +1)):
        v = np.array([r[key] for r in rows]) * sign
        if v.std() < 1e-12:
            continue
        parts = []
        for q in (0.0, 0.2, 0.4, 0.6, 0.8):
            t = float(np.quantile(v, q))
            m = v >= t
            parts.append(f"{name}>={t:.2f}: {ok[m].mean()*100:.0f}%/{int(m.sum())}")
        print(f"  {name} 임계별 정밀도/반환수:  " + "   ".join(parts))
    return out


def print_reloc_table(rows: list[dict]) -> None:
    print(f"  {'게이트':12s} {'질의':>5s} {'노드부족':>8s} {'반환':>5s} {'정답':>5s} "
          f"{'재현율':>7s} {'정밀도':>7s} {'중앙오차':>9s} {'중앙회전':>8s} "
          f"{'음성후보':>8s} {'음성반환':>8s} {'외부오탐':>8s}")
    for s in rows:
        print(f"  {s['gate']:12s} {s['n_query']:5d} {s['silent_few_nodes']:8d} "
              f"{s['accepted']:5d} {s['correct']:5d} "
              f"{s['recall']*100:6.1f}% {s['precision']*100:6.1f}% "
              f"{s['med_trans']:8.3f}m {s['med_rot']:7.2f}° "
              f"{s['neg_nodes_ok']:8d} {s['neg_accepted']:8d} "
              f"{s['fp_outmap']*100:7.1f}%")


# --- 실행 ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--trans-thresh", type=float, default=0.25,
                    help="정답으로 인정할 병진 오차 상한 (m)")
    ap.add_argument("--rot-thresh", type=float, default=15.0,
                    help="정답으로 인정할 회전 오차 상한 (deg)")
    ap.add_argument("--min-nodes", type=int, default=4)
    ap.add_argument("--score-ratio", type=float, default=0.85,
                    help="score-ratio 게이트 임계 (C++ query() 하드코딩 값과 동일)")
    ap.add_argument("--agree", type=float, default=0.50,
                    help="pose-agree 게이트: 1·2위 포즈 거리 상한 (m)")
    ap.add_argument("--tight-trans", type=float, default=0.10,
                    help="엄격 임계 병진 (m). 느슨한 임계는 궤적 크기를 잰다")
    ap.add_argument("--tight-rot", type=float, default=10.0)
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="채점 시 추가로 걸 신뢰도 하한 (C++ 결정과 직교)")
    args = ap.parse_args()

    print(f"판정 기준: 병진 {args.trans_thresh:.2f} m 이내 AND 회전 "
          f"{args.rot_thresh:.1f}° 이내를 '정답' 으로 센다.")
    print(f"엄격 기준도 같이 낸다: 병진 {args.tight_trans:.2f} m / 회전 "
          f"{args.tight_rot:.1f}°. 느슨한 임계에서는 상수 추정기도 통과한다.")
    print("외부오탐 = 지도에 없는 시퀀스의 질의 중 노드가 충분한 것들 대비 "
          "무언가를 반환한 비율. 정답은 침묵이다.\n")

    summary: list[tuple[str, dict]] = []
    tight_summary: list[tuple[str, dict]] = []
    baselines: dict[str, tuple[float, float]] = {}
    for p in args.prefixes:
        prefix = Path(p)
        name = prefix.name
        frames = read_csv(prefix.with_name(prefix.name + "_frames.csv"))
        queries = read_csv(prefix.with_name(prefix.name + "_queries.csv"))
        places = read_csv(prefix.with_name(prefix.name + "_places.csv"))
        selft = read_csv(prefix.with_name(prefix.name + "_selftest.csv"))

        print("=" * 100)
        print(f"[{name}]  등록 장소 {len(places)}")
        if not frames:
            print("  프레임 기록이 없다. 실행 자체가 실패했다.")
            continue
        for rows_, label in ((frames, "frames"), (queries, "queries"),
                             (places, "places"), (selft, "selftest")):
            check_widths(rows_, label)

        print("\n1) 작동 범위 - 장면에 성좌를 이룰 객체가 있는가")
        report_envelope(frames, args.min_nodes)

        # 다중 프레임 질의가 실제로 성좌를 키웠는지. 이걸 안 보면 창을 늘렸는데
        # 노드가 그대로인 경우(모두 같은 객체로 병합)를 "효과 없음"으로 오독한다.
        qpos = [r for r in queries if r.get("tag") == "pos"]
        if qpos:
            single = np.array([inum(r, "n_nodes") for r in qpos])
            win = np.array([query_nodes(r) for r in qpos])
            print(f"  질의 성좌 크기: 단일프레임 평균 {single.mean():.2f} -> "
                  f"실제 질의 평균 {win.mean():.2f}   "
                  f"min_nodes 충족 {int((win >= args.min_nodes).sum())}/{len(win)}")

        print("\n2) 자기질의 - 측정기가 살아 있는가")
        alive = report_selftest(selft)

        print("\n3) 재지역화")
        if not places:
            print("  등록된 장소가 없다. 재지역화는 시도조차 되지 않았다.")
            continue
        if not alive:
            print("  (자기질의 미통과 - 아래 수치는 참고용)")
        b = report_baseline(queries, places, args.trans_thresh, args.rot_thresh)
        bt = report_baseline(queries, places, args.tight_trans, args.tight_rot,
                             quiet=True)
        if b is not None and bt is not None:
            baselines[name] = (b, bt)
        report_confidence(queries, args.trans_thresh, args.rot_thresh)

        print(f"  [느슨한 임계 {args.trans_thresh:.2f} m / {args.rot_thresh:.0f}°]")
        rows = [report_reloc(queries, g, args.trans_thresh, args.rot_thresh,
                             args.min_nodes, args.score_ratio, args.agree, args.min_conf)
                for g in GATES]
        print_reloc_table(rows)

        print(f"  [엄격 임계 {args.tight_trans:.2f} m / {args.tight_rot:.0f}°]")
        trows = [report_reloc(queries, g, args.tight_trans, args.tight_rot,
                              args.min_nodes, args.score_ratio, args.agree, args.min_conf)
                 for g in GATES]
        print_reloc_table(trows)

        summary.append((name, rows[GATES.index("cpp")]))
        tight_summary.append((name, trows[GATES.index("cpp")]))
        print()

    if summary:
        for title, table, tt, rt in (
                (f"느슨한 임계 {args.trans_thresh:.2f} m / {args.rot_thresh:.0f}°",
                 summary, args.trans_thresh, args.rot_thresh),
                (f"엄격 임계 {args.tight_trans:.2f} m / {args.tight_rot:.0f}°",
                 tight_summary, args.tight_trans, args.tight_rot)):
            print("=" * 100)
            print(f"요약 (C++ query() 결정, {title})")
            print(f"  {'시퀀스':14s} {'재현율':>7s} {'정밀도':>7s} {'내부오탐':>8s} "
                  f"{'외부오탐':>8s} {'중앙오차':>9s} {'상수기준선':>10s}")
            for name, s in table:
                idx = 0 if table is summary else 1
                base = baselines.get(name)
                bstr = f"{base[idx]*100:9.1f}%" if base else f"{'-':>10s}"
                print(f"  {name:14s} {s['recall']*100:6.1f}% {s['precision']*100:6.1f}% "
                      f"{s['fp_inmap']*100:7.1f}% {s['fp_outmap']*100:7.1f}% "
                      f"{s['med_trans']:8.3f}m {bstr}")
            print("  상수기준선 = '항상 지도 중심 포즈' 추정기의 임계 통과율. "
                  "재현율이 이보다 낮으면 알고리즘은 아무것도 벌지 못한 것이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
