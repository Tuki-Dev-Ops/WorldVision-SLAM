#!/usr/bin/env python3
"""benchmark.json -> results/bench/viewer.tsv (네이티브 뷰어용 매니페스트).

왜 별도 파일인가
----------------
`wme_bench_viewer.exe` 가 지표를 **다시 계산하면 안 된다.** 다시 계산하는 순간
뷰어의 숫자와 문서의 숫자가 갈릴 수 있고, 그러면 어느 쪽이 맞는지 아무도
모른다. 06-results.md 가 반복해 적어 둔 규칙이다 - 추정과 채점은 분리하되,
채점은 한 곳에서만 한다.

그래서 뷰어는 궤적 파일(TUM 형식)과 영상만 읽고, **ATE/RPE/속도는 여기서
받아 간다.** 이 파일이 없으면 뷰어는 실행을 거부한다 (조용히 0 을 그리지 않는다).

형식: 탭 구분, '#' 은 주석.
  SEQ   <name> <dataset> <seq_dir> <gt_file> <identity_ate_cm>
  RUN   <name> <system> <label> <kind> <ate_cm> <rpe_mm> <ms_per_frame>
        <frames> <status> <traj_file>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def fmt(v, nd=4) -> str:
    if v is None:
        return "nan"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "nan"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=ROOT / "results" / "bench" / "benchmark.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "bench" / "viewer.tsv")
    args = ap.parse_args()

    if not args.bench.exists():
        raise SystemExit(f"{args.bench} 가 없다 - 먼저 bench_run.py 를 돌려야 한다")

    report = json.loads(args.bench.read_text(encoding="utf-8"))
    systems = report.get("systems", {})
    out = args.out
    rows: list[str] = []
    rows.append("# WorldVision-SLAM 벤치마크 뷰어 매니페스트")
    rows.append("# 지표는 python/tools/bench_run.py 가 계산한 값 그대로다.")
    rows.append("# SEQ\tname\tdataset\tseq_dir\tgt_file\tidentity_ate_cm")
    rows.append("# RUN\tname\tsystem\tlabel\tkind\tate_cm\trpe_mm\tms_per_frame"
                "\tframes\tstatus\ttraj_file")
    rows.append("# ERR\tname\tsystem\t<per-frame ATE error, cm, thinned to <=512>")

    n_seq = n_run = 0
    for e in report.get("sequences", []):
        name = e["name"]
        dataset = e.get("dataset", "tum")
        # TUM 만 폴더 이름에 접두사가 붙는다 (data/rgbd_dataset_<name>).
        # 나머지는 시퀀스 이름이 곧 폴더 이름이다.
        seq_dir = (ROOT / "data"
                   / (f"rgbd_dataset_{name}" if dataset == "tum" else name))
        gt = seq_dir / "groundtruth.txt"
        rows.append("\t".join([
            "SEQ", name, dataset, seq_dir.as_posix(), gt.as_posix(),
            fmt(e.get("identity_ate_cm"), 2)]))
        n_seq += 1

        for key, run in e.get("runs", {}).items():
            if not run.get("ok"):
                continue
            meta = systems.get(key, {})
            traj = out.parent / f"{name}_{key}.txt"
            rows.append("\t".join([
                "RUN", name, key,
                meta.get("label", key), meta.get("kind", "?"),
                fmt(run.get("ate_rmse_cm"), 2),
                fmt(run.get("rpe_trans_median_mm"), 2),
                fmt(run.get("ms_per_frame"), 1),
                str(run.get("frames", 0)),
                run.get("status", "ok"),
                traj.as_posix()]))
            n_run += 1

            # 프레임별 ATE 오차 계열. 두 시스템의 점군은 짧은 구간에서 거의
            # 같아 보이므로, **언제** 갈라지는지는 이 계열로만 보인다.
            # 값 자체는 bench_run.py 가 계산한 것을 그대로 옮긴다.
            errs = run.get("ate_errors_cm") or []
            if errs:
                # 화면 폭이 1000 px 이 안 되므로 512 점으로 줄인다. 최대값을
                # 유지하도록 구간 최대로 뽑는다 - 평균을 쓰면 스파이크가 사라지고,
                # 스파이크가 바로 이 그래프가 보여 줘야 하는 것이다.
                k = max(1, len(errs) // 512)
                thin = [max(errs[i:i + k]) for i in range(0, len(errs), k)]
                rows.append("\t".join(["ERR", name, key]
                                      + [f"{v:.3f}" for v in thin]))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"저장: {out}  ({n_seq} 시퀀스 / {n_run} 실행)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
