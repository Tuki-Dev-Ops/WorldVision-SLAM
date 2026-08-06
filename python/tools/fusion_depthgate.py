"""Tier 0 의 정보행렬을 기하 신뢰도로 줄이면 융합이 나아지는가.

25.6 이 남긴 마지막 항목이다: 신호는 계산되고 소비(강등)되지만, **아래쪽에서
아무도 그 신뢰도를 읽지 않는다.** 가장 자연스러운 소비처는 융합이다.

논거는 18.2 에 있다: *"좋은 추정과 나쁜 추정을 정보로 가중해 섞으면, 나쁜 쪽의
정보가 정직하게 작지 않은 한 좋은 쪽이 나빠진다."* Tier 0 는 23.3 에서 바닥값의
11.6 배로 표류하면서 자기 정보를 전혀 줄이지 않았다. depthReliability() 는 그
"정직하게 작아지는" 경로를 측광 밖에서 제공한다.

**반대 근거도 분명하다.** Tier 0 를 낮추면 그 무게는 3~15 배 부정확한 tier 로
넘어간다(18.2). 그래서 이것은 개선을 주장하는 실험이 아니라 **어느 쪽인지 재는**
실험이다.

비교는 21절의 재생 하네스 위에서 한다 - C++ 궤적 40 개를 0.002 % 로 재현한 그
포트다. 따라서 여기서 바뀌는 것은 오직 Tier 0 의 가중뿐이다.

사용:
  python tools/fusion_depthgate.py <data-root> <results-dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fusion_alpha import Seq, load_all  # noqa: E402
from fusion_replay import fuse_frames  # noqa: E402


def w_plain(mask):
    return lambda r: tuple(1.0 if mask[k] else 0.0 for k in range(3))


def w_depth_weighted(mask, floor: float = 0.0):
    """Tier 0 만 기하 신뢰도로 줄인다. 다른 tier 는 건드리지 않는다 - 건드리면
    무엇이 변화를 만들었는지 말할 수 없다."""
    def fn(r):
        w = [1.0 if mask[k] else 0.0 for k in range(3)]
        if w[0] > 0.0:
            w[0] *= max(floor, r.depth_rel)
        return tuple(w)
    return fn


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    seqs = load_all(Path(sys.argv[1]), Path(sys.argv[2]))
    if not seqs:
        print("tiers.csv 없음")
        return 2

    # 판별 가드: 기록에 신뢰도가 실제로 들어 있는가. 전부 1.0 이면 두 구성이
    # 같은 것이 되고, 그때 "차이 없음" 은 결과가 아니라 측정 실패다 (10.4).
    print(f"{'sequence':<24}{'frames':>7}{'judged':>8}{'rel<1':>7}{'mean rel':>10}")
    live = 0
    for s in seqs:
        rel = np.array([r.depth_rel for r in s.records if r.ref_idx >= 0])
        inc = np.array([r.depth_incons for r in s.records if r.ref_idx >= 0])
        judged = int((inc >= 0).sum())
        below = int((rel < 1.0).sum())
        live += below > 0
        print(f"{s.name:<24}{len(rel):>7}{judged:>8}{below:>7}{rel.mean():>10.3f}")
    if live == 0:
        print("\n신뢰도가 전부 1.0 이다 - 기록에 t0_depth_rel 이 없거나 게이트가")
        print("한 번도 발동하지 않았다. 이 상태의 비교는 아무 것도 재지 못한다.")
        return 1

    configs = [
        ("t0 only", (True, False, False)),
        ("t0+t2", (True, False, True)),
        ("all 3", (True, True, True)),
    ]
    print("\n" + "=" * 82)
    print("ATE cm — uniform weights vs Tier-0 scaled by depthReliability()")
    print("=" * 82)
    print(f"{'sequence':<24}" + "".join(f"{n:>18}" for n, _ in configs))
    print(f"{'':<24}" + "".join(f"{'plain / gated':>18}" for _ in configs))

    wins = losses = ties = 0
    for s in seqs:
        cells = ""
        for _, mask in configs:
            a = s.ate(fuse_frames(s.records, w_plain(mask)))
            b = s.ate(fuse_frames(s.records, w_depth_weighted(mask)))
            rel = (b - a) / max(a, 1e-9)
            if abs(rel) < 0.02:
                ties += 1
            elif rel < 0:
                wins += 1
            else:
                losses += 1
            cells += f"{a:>8.2f} /{b:>8.2f}"
        print(f"{s.name:<24}{cells}")

    print(f"\ngated better on {wins}, worse on {losses}, tied on {ties} "
          f"(of {wins + losses + ties} configurations)")
    print("\n2 % 미만 차이는 동점으로 센다. 21.3 에서 20.72 -> 20.72 를 '이겼다'로")
    print("세면 사실상 아무 것도 안 한 설정이 승리로 기록되는 것을 봤다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
