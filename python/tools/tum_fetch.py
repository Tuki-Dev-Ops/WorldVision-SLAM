"""TUM 시퀀스에서 시간 연속 구간만 스트리밍으로 받아 푼다.

아카이브를 디스크에 저장하지 않는다. HTTP 응답을 그대로 tarfile 스트림으로
열어 필요한 멤버만 쓴다 - 400~700 MB 를 저장할 공간 없이 원하는 구간만 얻는다.

시간 연속이 요점이다. tar 안의 파일 순서는 시각 순이 아니라서, "앞에서부터
N 개" 로 뽑으면 26 초 구간에 듬성듬성 흩어진 프레임이 나온다. 그 상태로
오도메트리를 돌리면 프레임 간 이동이 수렴 반경을 넘어, 엔진이 아니라 데이터
때문에 실패한다. 실제로 한 번 그렇게 측정했다 (docs/06-results.md 12.2).

스트림은 되감을 수 없으므로 무엇을 남길지 미리 알아야 한다. TUM 이 따로
제공하는 groundtruth.txt(0.2 MB)에서 시작 시각을 얻고, 영상 파일명 자체가
타임스탬프이므로 한 번 훑으면서 창 안의 것만 쓴다.

사용: python tools/tum_fetch.py <시퀀스이름> [--data DIR] [--seconds S] [--skip S]
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://cvg.cit.tum.de/rgbd/dataset"


def _group_of(name: str) -> str:
    for g in ("freiburg1", "freiburg2", "freiburg3"):
        if g in name:
            return g
    raise ValueError(f"freiburg 그룹을 알 수 없다: {name}")


def _stamp_of(member_name: str) -> float | None:
    """rgb/1305031102.175304.png -> 1305031102.175304"""
    try:
        return float(Path(member_name).stem)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="예: rgbd_dataset_freiburg3_walking_xyz")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--seconds", type=float, default=9.0, help="추출할 구간 길이")
    ap.add_argument("--skip", type=float, default=0.0, help="시작에서 건너뛸 시간")
    args = ap.parse_args()

    name = args.name
    group = _group_of(name)
    data = Path(args.data)
    root = data / name
    root.mkdir(parents=True, exist_ok=True)

    # 1) 시작 시각. groundtruth 는 따로 받을 수 있어 아카이브를 열기 전에 알 수 있다.
    gt_url = f"{BASE}/{group}/{name}-groundtruth.txt"
    req = urllib.request.Request(gt_url, headers={"User-Agent": "wme/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        gt_text = r.read().decode("utf-8", "replace")
    stamps = [float(line.split()[0]) for line in gt_text.splitlines()
              if line.strip() and not line.startswith("#")]
    if not stamps:
        print(f"groundtruth 를 읽지 못했다: {gt_url}", file=sys.stderr)
        return 1

    t0 = min(stamps) + args.skip
    t1 = t0 + args.seconds
    print(f"{name}: 창 {t0:.2f} ~ {t1:.2f} ({args.seconds:.1f} s)")

    # 2) 아카이브를 한 번 훑으며 창 안의 영상과 모든 txt 를 쓴다.
    url = f"{BASE}/{group}/{name}.tgz"
    got_img = got_txt = 0
    written = 0
    req = urllib.request.Request(url, headers={"User-Agent": "wme/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        # r|gz = 스트리밍 모드. seek 없이 앞에서부터 한 번만 훑는다.
        with tarfile.open(fileobj=io.BufferedReader(resp, 1 << 20), mode="r|gz") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                parts = m.name.split("/")
                if len(parts) < 2:
                    continue
                rel = "/".join(parts[1:])

                if rel.endswith(".txt"):
                    keep = True
                elif parts[1] in ("rgb", "depth"):
                    s = _stamp_of(rel)
                    keep = s is not None and t0 <= s <= t1
                else:
                    keep = False
                if not keep:
                    continue

                dst = root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(m)
                if src is None:
                    continue
                with open(dst, "wb") as out:
                    while chunk := src.read(1 << 20):
                        out.write(chunk)
                        written += len(chunk)
                if rel.endswith(".txt"):
                    got_txt += 1
                else:
                    got_img += 1
                    if got_img % 200 == 0:
                        print(f"  영상 {got_img}  {written/1e6:.0f} MB")

    print(f"완료: 영상 {got_img}, txt {got_txt}, {written/1e6:.1f} MB")
    return 0 if got_img > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
