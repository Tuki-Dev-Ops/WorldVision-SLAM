"""Build the self-contained side-by-side benchmark viewer.

Reads `results/bench/benchmark.json` (produced by `bench_run.py`) and emits a
single HTML file with the data inlined - no server, no network, no build step.
Open it, or publish it.

Layout is fixed by what the comparison is: **left is the classical descriptor
pipeline, right is WME.** Every panel keeps that side.

The trajectories are Umeyama-aligned to ground truth before display, using the
same alignment the ATE is scored with - otherwise the picture and the number
would disagree. The 2D projection is the ground truth's own two principal axes,
computed per sequence and applied unchanged to every system, so no system gets a
flattering viewpoint and no sequence gets a degenerate one.

Usage:
  python tools/bench_report.py [--in results/bench/benchmark.json]
                               [--out results/bench/index.html]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wme.eval.trajectory import (  # noqa: E402
    Trajectory, associate, evaluate_ate, umeyama)
from wme.reference.geometry import SE3  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# 검증된 팔레트의 첫 세 슬롯. 궤적 그림은 all-pairs 이므로 세 개까지가 상한이고,
# 진리값은 계열이 아니라 기준선이라 muted ink 로 따로 뺀다.
SERIES_ORDER = ["baseline", "wme", "wme_masked"]


def cv2_control_ate(name: str, gt: Trajectory) -> float | None:
    """제3자 구현 대조군(cv2.Odometry)의 ATE.

    자체 구현 대조군은 "덜 튜닝된 것 아니냐" 는 반론에서 자유롭지 않다. 이
    저장소와 아무 관련 없는 공개 구현이 같은 자리에 떨어지면 그 반론이 닫힌다.
    추정은 `tools/baseline_cv2.py` 가 이미 해 두었고 여기서는 채점만 한다.
    """
    from wme.eval.tum import load_trajectory
    p = ROOT / "results" / "bench" / f"{name}_cv2rgbd.txt"
    if not p.exists():
        return None
    try:
        return float(evaluate_ate(load_trajectory(p), gt, align=True).rmse) * 100.0
    except Exception as e:
        # 삼키지 않는다. 조용히 None 을 돌려주면 대조군 열이 통째로 사라지고,
        # 표는 "제3자 대조군을 안 돌렸다" 와 똑같이 보인다 (19.5 의 그 줄).
        print(f"  [cv2 control] {name}: {type(e).__name__}: {e}")
        return None


def project(seq: dict) -> dict:
    """Align every system to ground truth, then project onto GT's principal axes."""
    gt = seq.get("gt_traj") or []
    gt_xyz = np.array([p for p in gt if p is not None], dtype=float)
    if len(gt_xyz) < 3:
        return {}

    # 기저는 진리값에서만 뽑는다. 시스템별로 다시 뽑으면 각자에게 유리한
    # 시점이 주어져 그림이 비교가 아니게 된다.
    c = gt_xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(gt_xyz - c, full_matrices=False)
    basis = vt[:2]

    def to2d(xyz: np.ndarray) -> list[list[float]]:
        return [[round(float(v[0]), 4), round(float(v[1]), 4)]
                for v in (xyz - c) @ basis.T]

    out = {"gt": to2d(gt_xyz)}
    gt_stamps = np.array([s for s, p in zip(seq["_gt_stamps"], gt) if p is not None])
    gt_traj = Trajectory(gt_stamps, [SE3(np.eye(3), p) for p in gt_xyz])

    for key in SERIES_ORDER:
        run = seq["runs"].get(key)
        if not run or not run.get("ok"):
            continue
        est = Trajectory(np.array(run["stamps"]),
                         [SE3(np.eye(3), np.array(p)) for p in run["traj"]])
        e, g = associate(est, gt_traj)
        if len(e) < 3:
            continue
        # evaluate_ate 와 **같은** 정렬이어야 한다: 같은 연관, 같은 with_scale.
        # 다르게 맞추면 그림과 숫자가 조용히 어긋난다.
        R, t, s = umeyama(e.positions, g.positions, False)
        aligned = (s * (np.array(run["traj"]) @ R.T)) + t
        out[key] = to2d(aligned)
    return out


def compact(report: dict) -> dict:
    """Trim the run record to what the page draws, and round it."""
    seqs = []
    for s in report["sequences"]:
        s["_gt_stamps"] = s["runs"][next(iter(s["runs"]))]["stamps"]
        paths = project(s)
        runs = {}
        for key in SERIES_ORDER:
            r = s["runs"].get(key)
            if not r or not r.get("ok"):
                continue
            rel = r.get("rel", {})
            runs[key] = {
                "status": r.get("status", "ok"),
                "ate": round(r["ate_rmse_cm"], 3),
                "rpe": round(r["rpe_trans_median_mm"] or 0.0, 3),
                "rpe95": round(r["rpe_trans_p95_mm"] or 0.0, 3),
                "rot": round(r["rpe_rot_median_deg"] or 0.0, 4),
                # None 을 0 으로 접으면 "0 ms/frame", 즉 무한히 빠르다는 주장이
                # 된다. 재채점(--skip-run)에는 벽시계가 없으므로 없음을 유지한다.
                "ms": (round(r["ms_per_frame"], 2)
                       if r.get("ms_per_frame") is not None else None),
                "frames": r["frames"],
                "lost": r.get("track_lost"),
                "path": paths.get(key, []),
                # 프레임별 오차 곡선. 시간축은 시퀀스 시작 기준 상대초로 둔다.
                "t": [round(x - rel["stamps"][0], 3) for x in rel.get("stamps", [])],
                "err": [round(x * 1000.0, 3) for x in rel.get("trans", [])],
                "diag": {k: (round(v, 3) if isinstance(v, (int, float)) else v)
                         for k, v in (r.get("diag_extra") or {}).items()
                         if v is not None},
            }
        gts = np.array(s["_gt_stamps"])
        gtp = [p for p in s.get("gt_traj", []) if p is not None]
        cv2_ate = None
        if gtp:
            st = np.array([t for t, p in zip(s["_gt_stamps"], s["gt_traj"])
                           if p is not None])
            gt_traj = Trajectory(st, [SE3(np.eye(3), np.array(p)) for p in gtp])
            cv2_ate = cv2_control_ate(s["name"], gt_traj)
        entry = {
            "name": s["name"],
            "identity": round(s["identity_ate_cm"], 3),
            "gt": paths.get("gt", []),
            "runs": runs,
        }
        if cv2_ate is not None:
            entry["cv2"] = round(cv2_ate, 3)
        seqs.append(entry)
    return {"sequences": seqs, "systems": report["systems"]}


HTML = r"""<title>WME vs classical SLAM — side-by-side benchmark</title>
<style>
:root{
  color-scheme: light;
  --surface:#fcfcfb; --plane:#f9f9f7;
  --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --good:#0ca30c; --crit:#d03b3b;
}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme=light])){
  color-scheme: dark;
  --surface:#1a1a19; --plane:#0d0d0d;
  --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70;
}}
:root[data-theme=dark]{
  color-scheme: dark;
  --surface:#1a1a19; --plane:#0d0d0d;
  --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70;
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;}
.wrap{max-width:1240px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:27px;line-height:1.25;margin:0 0 8px;letter-spacing:-.015em}
h2{font-size:19px;margin:40px 0 6px;letter-spacing:-.01em}
h3{font-size:14px;margin:0;color:var(--ink2);font-weight:600}
p{color:var(--ink2);margin:6px 0 0;max-width:76ch}
.sub{font-size:13.5px;color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:18px}
.note{border-left:3px solid var(--s2);padding:10px 14px;background:var(--surface);
  border-radius:0 8px 8px 0;margin:16px 0;font-size:13.5px;color:var(--ink2)}
.legend{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:10px 0 14px;font-size:13px}
.lg{display:inline-flex;gap:7px;align-items:center;color:var(--ink2)}
.sw{width:13px;height:3px;border-radius:2px;display:inline-block}
.sw.d{height:0;border-top:2px dashed var(--muted);width:15px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.side{border:1px solid var(--ring);border-radius:12px;background:var(--surface);overflow:hidden}
.side header{padding:13px 16px;border-bottom:1px solid var(--ring);
  display:flex;justify-content:space-between;align-items:baseline;gap:10px}
.tag{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--ring)}
.tile{background:var(--surface);padding:11px 13px}
.tile .k{font-size:11px;color:var(--muted);letter-spacing:.03em}
.tile .v{font-size:21px;font-weight:600;letter-spacing:-.02em;margin-top:2px}
.tile .u{font-size:12px;color:var(--muted);font-weight:400}
select{font:inherit;padding:7px 11px;border-radius:8px;border:1px solid var(--ring);
  background:var(--surface);color:var(--ink)}
:focus-visible{outline:2px solid var(--s1);outline-offset:2px;border-radius:4px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
table{border-collapse:collapse;width:100%;font-size:13px;
  font-variant-numeric:tabular-nums}
th,td{padding:7px 9px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th:first-child,td:first-child{text-align:left;font-variant-numeric:normal}
th{color:var(--muted);font-weight:600;font-size:11.5px;letter-spacing:.03em;
  text-transform:uppercase}
tbody tr:hover{background:var(--plane)}
.win{color:var(--good);font-weight:600}
.lose{color:var(--crit)}
.scroll{overflow-x:auto}
svg{display:block;max-width:100%}
.tt{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);
  border:1px solid var(--ring);border-radius:8px;padding:7px 10px;font-size:12.5px;
  box-shadow:0 4px 14px rgba(0,0,0,.13);opacity:0;transition:opacity .1s;z-index:9;
  font-variant-numeric:tabular-nums}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--ring);
  font-size:13px;color:var(--muted)}
code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;
  background:var(--plane);padding:1px 5px;border-radius:4px}
</style>

<div class="wrap">
<h1>Descriptor-free vs. classical: a side-by-side SLAM benchmark</h1>
<p><strong>Left is the classical descriptor pipeline. Right is WME.</strong> Same frames, same
undistortion, intrinsics, depth scale and keyframe rule; scored by a third program that
estimated neither. The per-sequence panels are odometry only; loop closure is measured
separately, further down, with a back-end shared by both systems.</p>

<div class="note"><strong>Two controls, and neither is ORB-SLAM3.</strong>
<em>ORB + PnP</em> — Hamming descriptor matching (Lowe ratio 0.75) → RANSAC PnP → LM
refinement, 1000 features, 8 pyramid levels, ORB-SLAM3's own front-end settings — but with no
loop closure, local map or bundle adjustment. <em>cv2.Odometry</em> — OpenCV 5's published
dense RGB-D odometry, written by nobody here, run through the identical data path; it is the
evidence that the hand-written control is not under-tuned. Every ATE below is measured against
<strong>the better of the two</strong>, because one control flatters: against ORB alone the
tally is 12–4.</div>

<h2>Verdict</h2>
<div id="hero"></div>

<h2>Absolute trajectory error, all sequences</h2>
<p class="sub">Lower is better. The dashed rule is the “camera never moved” floor,
recomputed per sequence — anything at or above it has learned nothing.</p>
<div class="legend" id="lgd"></div>
<div class="card"><div id="bars"></div></div>

<h2>Per sequence</h2>
<div style="margin:10px 0 14px"><select id="seq"></select></div>
<div class="grid2" id="panels"></div>

<h2 style="margin-top:34px">Per-frame error over time</h2>
<p class="sub">Frame-to-frame relative translation error against ground truth —
the quantity neither system optimises directly.</p>
<div class="legend" id="lgd2"></div>
<div class="card"><div id="series"></div></div>

<div id="degrade-sec" hidden>
<h2>Under degradation</h2>
<p class="sub">Haze applied to the real frames by the scattering equation
<code>I·t + A·(1−t)</code>, with <code>t = exp(−β·d)</code> evaluated on TUM's
<strong>measured</strong> depth — so the transmission is not an approximation of scattering.
Measured contrast tracks the prediction to within 2%.</p>
<div class="legend" id="lgd3"></div>
<div class="card"><div id="degplot"></div></div>
<div class="note" id="degnote"></div>
</div>

<div id="loop-sec" hidden>
<h2>With loop closure</h2>
<p class="sub">Same keyframes, same candidate proposal, same pose graph, same robust kernel.
The only difference is how a proposed loop is verified.</p>
<div class="card scroll"><div id="loop"></div></div>
</div>

<h2>All numbers</h2>
<p class="sub">ATE cm · RPE median mm · ms per frame, end-to-end wall clock.</p>
<div class="card scroll"><div id="table"></div></div>

<footer id="foot"></footer>
</div>
<div class="tt" id="tt"></div>

<script>
const DATA = __DATA__;
const SY = {baseline:{label:"ORB + PnP (classical)",c:"var(--s1)",short:"ORB+PnP"},
            wme:{label:"WME ECDA (Tier 0)",c:"var(--s2)",short:"WME"},
            wme_masked:{label:"WME + token mask",c:"var(--s3)",short:"WME+mask"}};
const KEYS = ["baseline","wme","wme_masked"];
const tt = document.getElementById("tt");
const NS = "http://www.w3.org/2000/svg";
const el = (n,a={})=>{const e=document.createElementNS(NS,n);
  for(const k in a) e.setAttribute(k,a[k]); return e;};
const fmt = (v,d=2)=> v==null||!isFinite(v) ? "—" : v.toFixed(d);
/* 축 라벨용 축약. 회전 라벨 12개가 전부 "freiburg" 로 시작하면 서로 겹치고,
   겹치는 순간 공통 접두사는 정보가 아니라 방해물이다. */
const shortName = n => n.replace(/^freiburg(\d)_/, "fr$1 ").replace(/_/g, " ");

function showTip(ev,html){tt.innerHTML=html;tt.style.opacity=1;
  const r=tt.getBoundingClientRect();
  let x=ev.clientX+14, y=ev.clientY-10;
  if(x+r.width>innerWidth-8) x=ev.clientX-r.width-14;
  if(y+r.height>innerHeight-8) y=innerHeight-r.height-8;
  tt.style.left=x+"px"; tt.style.top=Math.max(8,y)+"px";}
function hideTip(){tt.style.opacity=0;}

/* 실패한 실행의 ATE 는 숫자로 취급하지 않는다. 항등 궤적은 do-nothing 바닥값과
   같은 점수를 받고(즉 표에서 선전하는 것처럼 보이고), 발산한 실행은 정렬 뒤에도
   유한한 값을 내지만 그 값은 아무 것도 뜻하지 않는다. */
const isOk = r => r && (r.status||"ok")==="ok";
const STATUS_LABEL = {diverged:"diverged", no_output:"no output"};

/* best WME variant per sequence = the system as shipped, choosing by ATE */
function bestWme(s){
  let b=null;
  for(const k of ["wme","wme_masked"]){const r=s.runs[k];
    if(isOk(r) && (!b || r.ate<b.r.ate)) b={k,r};}
  return b;
}

/* ---------- verdict ---------- */
(function(){
  // 승패는 **대조군 둘 중 더 좋은 쪽**과 비교한다. 대조군 하나만 놓고 세면
  // 결과가 후해진다 - 실제로 ORB 하나로는 12-4 였고, 공개 구현을 하나 더
  // 넣자 WME 수치는 하나도 바뀌지 않은 채 9-6 이 됐다.
  let w=0,l=0, rb=[], rw=[], failed=0;
  for(const s of DATA.sequences){
    const b=s.runs.baseline, bw=bestWme(s);
    if(!bw || (!isOk(b) && s.cv2==null)){ failed++; continue; }
    const ext=Math.min(isOk(b)?b.ate:Infinity, s.cv2!=null?s.cv2:Infinity);
    if(!isFinite(ext)){ failed++; continue; }
    if(bw.r.ate<ext) w++; else l++;
    if(isOk(b)){ rb.push(b.rpe); rw.push(bw.r.rpe); }
  }
  const med=a=>{const x=[...a].sort((p,q)=>p-q);const n=x.length;
    return n?(n%2?x[(n-1)/2]:(x[n/2-1]+x[n/2])/2):NaN;};
  let rpeWin=0;
  for(const s of DATA.sequences){const b=s.runs.baseline,bw=bestWme(s);
    if(isOk(b)&&bw&&bw.r.rpe<b.rpe) rpeWin++;}
  document.getElementById("hero").innerHTML=`
   <div class="grid2">
    <div class="card"><h3>Absolute trajectory error (ATE)</h3>
      <div style="font-size:34px;font-weight:600;letter-spacing:-.03em;margin:6px 0 2px">
        ${w} <span style="font-size:17px;color:var(--muted);font-weight:400">of ${w+l} to WME,
        vs the better of two controls</span></div>
      <p class="sub" style="margin-top:4px">Against the ORB control alone it is 12–4. Adding a
      second, published implementation moved it here without a single WME number changing.
      ${failed?`${failed} sequence${failed>1?'s':''} unscorable — a system diverged or produced
      nothing, and winning against a failure is not a result.`:''}</p></div>
    <div class="card"><h3>Per-frame accuracy (RPE)</h3>
      <div style="font-size:34px;font-weight:600;letter-spacing:-.03em;margin:6px 0 2px">
        ${rpeWin} <span style="font-size:17px;color:var(--muted);font-weight:400">of ${w+l} sequences to WME</span></div>
      <p class="sub" style="margin-top:4px">Median ${fmt(med(rw))} mm vs ${fmt(med(rb))} mm.
      WME is more accurate frame-to-frame almost everywhere — yet loses ATE on a third of them.</p></div>
   </div>
   <div class="note" style="border-color:var(--s1)"><strong>That gap is the finding.</strong>
   Smaller per-frame error with larger accumulated drift means WME's residual error is more
   <em>signed</em> than the baseline's, and only the signed part survives a pose chain.
   Magnitude and sign are different statistics, and ATE only reads one of them.</div>`;
})();

/* ---------- legends ---------- */
function legend(id, keys, extra){
  const parts = keys.map(k=>`<span class="lg"><span class="sw" style="background:${SY[k].c}"></span>${SY[k].label}</span>`);
  if(extra) parts.push(extra);
  document.getElementById(id).innerHTML = parts.join("");
}
legend("lgd",KEYS,
  `<span class="lg"><span class="sw" style="background:var(--ink2)"></span>cv2.Odometry — third-party control</span>`
  + `<span class="lg"><span class="sw d"></span>do-nothing floor</span>`);

/* ---------- grouped bars: ATE per sequence ---------- */
(function(){
  const S=DATA.sequences, W=Math.max(760, S.length*88), H=360;
  const m={t:14,r:14,b:96,l:52};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  // 실패한 실행은 축척 계산에서 뺀다. 발산 한 건(1e13 cm)이 들어오면 나머지
  // 막대가 전부 0 높이로 눌린다.
  const max=Math.max(...S.flatMap(s=>[...KEYS.map(k=>isOk(s.runs[k])?s.runs[k].ate:0),
                                      s.identity]));
  const y=v=>ih-Math.min(v,max)/max*ih;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H,role:"img",
    "aria-label":"Absolute trajectory error per sequence"});
  const g=el("g",{transform:`translate(${m.l},${m.t})`}); svg.appendChild(g);

  for(let i=0;i<=4;i++){const v=max*i/4;
    g.appendChild(el("line",{x1:0,x2:iw,y1:y(v),y2:y(v),stroke:"var(--grid)","stroke-width":1}));
    const t=el("text",{x:-9,y:y(v)+4,"text-anchor":"end",fill:"var(--muted)","font-size":11});
    t.textContent=v.toFixed(0); g.appendChild(t);}
  const yl=el("text",{transform:`translate(-38,${ih/2}) rotate(-90)`,"text-anchor":"middle",
    fill:"var(--muted)","font-size":11}); yl.textContent="ATE RMSE (cm)"; g.appendChild(yl);

  const bw=iw/S.length, gap=2, inner=Math.min(58,bw-16), bar=(inner-gap*2)/3;
  S.forEach((s,i)=>{
    const x0=i*bw+(bw-inner)/2;
    // do-nothing floor
    g.appendChild(el("line",{x1:i*bw+6,x2:(i+1)*bw-6,y1:y(s.identity),y2:y(s.identity),
      stroke:"var(--muted)","stroke-width":1.5,"stroke-dasharray":"4 3"}));
    // 제3자 대조군은 **막대가 아니라 기준 표식**으로 그린다. 계열을 넷으로
    // 늘리면 검증된 3슬롯 팔레트를 벗어나고(노랑이 주황 옆에 서면 all-pairs
    // 문턱을 통과하지 못한다), 애초에 이것은 비교 대상이 아니라 기준이다.
    if(s.cv2!=null && s.cv2<=max){
      const yy=y(s.cv2);
      g.appendChild(el("line",{x1:x0-3,x2:x0+inner+3,y1:yy,y2:yy,
        stroke:"var(--ink2)","stroke-width":2}));
      const hit=el("rect",{x:x0-3,y:yy-5,width:inner+6,height:10,fill:"transparent"});
      hit.addEventListener("mousemove",e=>showTip(e,
        `<b>${s.name}</b><br>cv2.Odometry (third-party control)<br>ATE <b>${fmt(s.cv2)} cm</b>`));
      hit.addEventListener("mouseleave",hideTip);
      g.appendChild(hit);
    }
    KEYS.forEach((k,j)=>{
      const r=s.runs[k]; if(!r) return;
      const x=x0+j*(bar+gap);
      if(!isOk(r)){
        // 실패는 막대가 아니라 표식으로. 막대로 그리면 "아주 나쁜 값" 처럼
        // 읽히지만 실제로는 값이 없는 것이다.
        const mk=el("text",{x:x+bar/2,y:ih-6,"text-anchor":"middle",
          fill:"var(--crit)","font-size":13,"font-weight":"700"});
        mk.textContent="✕";
        mk.addEventListener("mousemove",e=>showTip(e,
          `<b>${s.name}</b><br>${SY[k].label}<br>
           <span style="color:var(--crit)"><b>${STATUS_LABEL[r.status]||r.status}</b></span><br>
           ATE is not meaningful for this run`));
        mk.addEventListener("mouseleave",hideTip);
        g.appendChild(mk);
        return;
      }
      const h=Math.max(2,ih-y(r.ate));
      const rect=el("rect",{x,y:y(r.ate),width:bar,height:h,rx:4,fill:SY[k].c,
        stroke:"var(--surface)","stroke-width":2});
      rect.addEventListener("mousemove",e=>showTip(e,
        `<b>${s.name}</b><br>${SY[k].label}<br>ATE <b>${fmt(r.ate)} cm</b><br>
         RPE ${fmt(r.rpe)} mm · ${fmt(r.ms,1)} ms/frame<br>
         <span style="color:var(--muted)">floor ${fmt(s.identity)} cm</span>`));
      rect.addEventListener("mouseleave",hideTip);
      g.appendChild(rect);
    });
    const lab=el("text",{transform:`translate(${i*bw+bw/2},${ih+13}) rotate(38)`,
      "text-anchor":"start",fill:"var(--muted)","font-size":10.5});
    lab.textContent=shortName(s.name); g.appendChild(lab);
  });
  g.appendChild(el("line",{x1:0,x2:iw,y1:ih,y2:ih,stroke:"var(--axis)","stroke-width":1}));
  const d=document.getElementById("bars"); d.style.overflowX="auto"; d.appendChild(svg);
})();

/* ---------- trajectory panel ---------- */
function pathFor(pts,sx,sy){return pts.map((p,i)=>(i?"L":"M")+sx(p[0]).toFixed(1)+" "+sy(p[1]).toFixed(1)).join(" ");}

function drawPanel(host,s,key){
  const r=s.runs[key];
  host.innerHTML="";
  const sec=document.createElement("div"); sec.className="side";
  const isBase = key==="baseline";
  sec.innerHTML=`<header><div><h3>${SY[key].label}</h3>
      <div class="tag">${isBase?"existing model":"this project"}</div></div>
      <span class="sw" style="background:${SY[key].c};width:26px;height:4px"></span></header>`;
  if(!r){sec.innerHTML+=`<div style="padding:22px;color:var(--muted)">no run</div>`;
    host.appendChild(sec);return;}

  const t=document.createElement("div"); t.className="tiles";
  const cmp=s.runs.baseline;
  const dv=(v,b,inv)=>{if(!cmp||key==="baseline"||b==null)return"";
    const better=inv?v>b:v<b; const p=Math.abs((v-b)/b*100);
    return ` <span class="${better?'win':'lose'}" style="font-size:12px">${better?'▼':'▲'}${p.toFixed(0)}%</span>`;};
  t.innerHTML=`
   <div class="tile"><div class="k">ATE RMSE</div><div class="v">${fmt(r.ate)}<span class="u"> cm</span>${dv(r.ate,cmp&&cmp.ate)}</div></div>
   <div class="tile"><div class="k">RPE median</div><div class="v">${fmt(r.rpe)}<span class="u"> mm</span>${dv(r.rpe,cmp&&cmp.rpe)}</div></div>
   <div class="tile"><div class="k">per frame</div><div class="v">${fmt(r.ms,0)}<span class="u"> ms</span></div></div>
   <div class="tile"><div class="k">frames</div><div class="v">${r.frames}<span class="u">${r.lost?` · ${r.lost} lost`:""}</span></div></div>`;
  sec.appendChild(t);

  // trajectory, GT-aligned, GT principal axes
  const W=560,H=330,m=26;
  const all=[...s.gt,...(r.path||[])];
  if(all.length){
    const xs=all.map(p=>p[0]), ys=all.map(p=>p[1]);
    const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
    const sc=Math.min((W-2*m)/Math.max(1e-6,x1-x0),(H-2*m)/Math.max(1e-6,y1-y0));
    const cx=(x0+x1)/2, cy=(y0+y1)/2;
    const sx=v=>W/2+(v-cx)*sc, sy=v=>H/2-(v-cy)*sc;
    const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%",role:"img",
      "aria-label":`${s.name} trajectory, ${SY[key].label}`});
    svg.appendChild(el("path",{d:pathFor(s.gt,sx,sy),fill:"none",stroke:"var(--muted)",
      "stroke-width":2,"stroke-dasharray":"5 4","stroke-linejoin":"round"}));
    svg.appendChild(el("path",{d:pathFor(r.path,sx,sy),fill:"none",stroke:SY[key].c,
      "stroke-width":2,"stroke-linejoin":"round","stroke-linecap":"round"}));
    if(r.path.length){
      svg.appendChild(el("circle",{cx:sx(r.path[0][0]),cy:sy(r.path[0][1]),r:4.5,
        fill:"var(--surface)",stroke:SY[key].c,"stroke-width":2}));
      const e=r.path[r.path.length-1];
      svg.appendChild(el("circle",{cx:sx(e[0]),cy:sy(e[1]),r:4.5,fill:SY[key].c,
        stroke:"var(--surface)","stroke-width":2}));
    }
    const cap=el("text",{x:m-8,y:H-8,fill:"var(--muted)","font-size":11});
    cap.textContent="top-down, ground-truth principal axes · hollow = start";
    svg.appendChild(cap);
    const box=document.createElement("div"); box.style.padding="6px 10px 10px";
    box.appendChild(svg); sec.appendChild(box);
  }
  host.appendChild(sec);
}

/* ---------- per-frame error series ---------- */
/* rolling median - 원자료는 프레임 단위로 튀어서 두 곡선이 겹치면 아무것도
   안 보인다. 평균이 아니라 중앙값을 쓴다: 스파이크 하나가 창 전체를 끌고
   가면 그건 매끄럽게 만든 게 아니라 다른 신호를 그린 것이다. */
function rollMedian(a,w){
  const h=w>>1, out=new Array(a.length);
  for(let i=0;i<a.length;i++){
    const x=a.slice(Math.max(0,i-h),Math.min(a.length,i+h+1)).sort((p,q)=>p-q);
    out[i]=x.length%2?x[(x.length-1)/2]:(x[x.length/2-1]+x[x.length/2])/2;
  }
  return out;
}

function drawSeries(s){
  const host=document.getElementById("series"); host.innerHTML="";
  // 좌우 패널이 비교하는 바로 그 둘만 그린다. 세 계열을 겹쳐 놓으면
  // 페이지가 말하는 비교와 그림이 말하는 비교가 달라진다.
  const bw0=bestWme(s);
  const runs=["baseline", bw0?bw0.k:"wme"]
    .filter(k=>s.runs[k]&&s.runs[k].err&&s.runs[k].err.length);
  if(!runs.length){host.textContent="no data";return;}
  const W=1180,H=300,m={t:12,r:14,b:34,l:52};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const tmax=Math.max(...runs.map(k=>Math.max(...s.runs[k].t)));
  // p99 로 자른다. 한 프레임의 튀는 값이 전체를 눌러 버리면 곡선이 안 보인다.
  const pool=runs.flatMap(k=>s.runs[k].err).sort((a,b)=>a-b);
  const vmax=Math.max(1e-6,pool[Math.floor(pool.length*0.99)]||1);
  const sx=v=>v/Math.max(1e-6,tmax)*iw, sy=v=>ih-Math.min(v,vmax)/vmax*ih;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%",role:"img",
    "aria-label":"per-frame relative translation error"});
  const g=el("g",{transform:`translate(${m.l},${m.t})`}); svg.appendChild(g);
  for(let i=0;i<=4;i++){const v=vmax*i/4;
    g.appendChild(el("line",{x1:0,x2:iw,y1:sy(v),y2:sy(v),stroke:"var(--grid)","stroke-width":1}));
    const t=el("text",{x:-9,y:sy(v)+4,"text-anchor":"end",fill:"var(--muted)","font-size":11});
    t.textContent=v.toFixed(0); g.appendChild(t);}
  const yl=el("text",{transform:`translate(-38,${ih/2}) rotate(-90)`,"text-anchor":"middle",
    fill:"var(--muted)","font-size":11}); yl.textContent="rel. trans. error (mm)"; g.appendChild(yl);
  const xl=el("text",{x:iw/2,y:ih+27,"text-anchor":"middle",fill:"var(--muted)","font-size":11});
  xl.textContent="seconds into sequence"; g.appendChild(xl);
  runs.forEach(k=>{
    const r=s.runs[k];
    // 원자료는 옅게, 이동중앙값은 진하게. 둘 다 남겨야 "매끄럽게 만든 것"이
    // 무엇을 감췄는지 독자가 직접 볼 수 있다.
    const raw=r.t.map((tv,i)=>(i?"L":"M")+sx(tv).toFixed(1)+" "+sy(r.err[i]).toFixed(1)).join(" ");
    g.appendChild(el("path",{d:raw,fill:"none",stroke:SY[k].c,"stroke-width":1,
      "stroke-linejoin":"round",opacity:.22}));
    const med=rollMedian(r.err,9);
    const sm=r.t.map((tv,i)=>(i?"L":"M")+sx(tv).toFixed(1)+" "+sy(med[i]).toFixed(1)).join(" ");
    g.appendChild(el("path",{d:sm,fill:"none",stroke:SY[k].c,"stroke-width":2,
      "stroke-linejoin":"round","stroke-linecap":"round"}));
  });
  g.appendChild(el("line",{x1:0,x2:iw,y1:ih,y2:ih,stroke:"var(--axis)","stroke-width":1}));

  // crosshair
  const cross=el("line",{y1:0,y2:ih,stroke:"var(--axis)","stroke-width":1,opacity:0});
  g.appendChild(cross);
  const hit=el("rect",{x:0,y:0,width:iw,height:ih,fill:"transparent"});
  hit.addEventListener("mousemove",e=>{
    const bb=svg.getBoundingClientRect();
    const px=(e.clientX-bb.left)*(W/bb.width)-m.l;
    const tv=Math.max(0,Math.min(tmax,px/iw*tmax));
    cross.setAttribute("x1",sx(tv)); cross.setAttribute("x2",sx(tv));
    cross.setAttribute("opacity",1);
    let rows="";
    runs.forEach(k=>{const r=s.runs[k];
      let bi=0,bd=1e9; for(let i=0;i<r.t.length;i++){const d=Math.abs(r.t[i]-tv);
        if(d<bd){bd=d;bi=i;}}
      rows+=`<div><span class="sw" style="background:${SY[k].c};display:inline-block;
        width:11px;height:3px;margin-right:6px;vertical-align:middle"></span>
        ${SY[k].short} <b>${fmt(r.err[bi])}</b> mm</div>`;});
    showTip(e,`<b>t = ${tv.toFixed(2)} s</b>${rows}`);
  });
  hit.addEventListener("mouseleave",()=>{hideTip();cross.setAttribute("opacity",0);});
  g.appendChild(hit);
  host.appendChild(svg);
}

/* ---------- table (also the relief for the low-contrast slot) ---------- */
(function(){
  const anyCv2 = DATA.sequences.some(s=>s.cv2!=null);
  let h=`<table><thead><tr><th>sequence</th>`;
  KEYS.forEach(k=>h+=`<th>${SY[k].short} ATE</th>`);
  if(anyCv2) h+=`<th>cv2.Odometry ATE</th>`;
  h+=`<th>floor</th>`;
  KEYS.forEach(k=>h+=`<th>${SY[k].short} RPE</th>`);
  KEYS.forEach(k=>h+=`<th>${SY[k].short} ms</th>`);
  h+=`</tr></thead><tbody>`;
  for(const s of DATA.sequences){
    h+=`<tr><td>${s.name}</td>`;
    // 행 전체(대조군 둘 + WME 둘)의 최솟값을 표시한다. 2자 비교의 승자를
    // 4열 표에 칠하면 더 좋은 값이 옆 칸에 있는데도 다른 칸이 초록이 된다.
    const rowVals=[...KEYS.filter(k=>isOk(s.runs[k])).map(k=>s.runs[k].ate)];
    if(s.cv2!=null) rowVals.push(s.cv2);
    const rowMin=rowVals.length?Math.min(...rowVals):null;
    KEYS.forEach(k=>{const r=s.runs[k];
      if(r && !isOk(r)){
        h+=`<td class="lose" title="ATE is not meaningful for this run">${STATUS_LABEL[r.status]||r.status}</td>`;
        return;
      }
      const best = isOk(r) && rowMin!=null && r.ate===rowMin;
      h+=`<td${best?' class="win"':''}>${r?fmt(r.ate):"—"}</td>`;});
    if(anyCv2) h+=`<td${(s.cv2!=null&&s.cv2===rowMin)?' class="win"':' style="color:var(--ink2)"'}>${s.cv2!=null?fmt(s.cv2):"—"}</td>`;
    h+=`<td style="color:var(--muted)">${fmt(s.identity)}</td>`;
    KEYS.forEach(k=>{const r=s.runs[k];h+=`<td>${r?fmt(r.rpe):"—"}</td>`;});
    KEYS.forEach(k=>{const r=s.runs[k];h+=`<td>${r?fmt(r.ms,0):"—"}</td>`;});
    h+=`</tr>`;
  }
  h+=`</tbody></table>`;
  document.getElementById("table").innerHTML=h;
})();

/* ---------- degradation sweep ---------- */
(function(){
  const D = DATA.degrade;
  if(!D || !D.sequences || !D.sequences.length) return;
  document.getElementById("degrade-sec").hidden = false;
  legend("lgd3",["baseline","wme"],
    `<span class="lg" style="color:var(--muted)">solid = fr1 xyz · dashed = fr1 desk</span>`);

  const W=1180,H=340,m={t:14,r:16,b:46,l:60};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const betas=D.betas;
  // ATE 가 1.5 cm 에서 5433 cm 까지 세 자릿수 넘게 움직인다. 선형 축이면
  // 아래쪽 절반이 전부 0 에 눌려 붙어 아무것도 안 보인다.
  const vals=[];
  D.sequences.forEach(s=>s.rows.forEach(r=>KEYS.forEach(k=>{
    const v=r.runs&&r.runs[k]; if(v&&v.ate>0) vals.push(v.ate);})));
  if(!vals.length) return;
  const lo=Math.max(0.5,Math.min(...vals)*0.7), hi=Math.max(...vals)*1.4;
  const sy=v=>ih-(Math.log10(Math.max(v,lo))-Math.log10(lo))/(Math.log10(hi)-Math.log10(lo))*ih;
  const sx=i=>i/(betas.length-1)*iw;

  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%",role:"img",
    "aria-label":"ATE versus haze"});
  const g=el("g",{transform:`translate(${m.l},${m.t})`}); svg.appendChild(g);
  for(let e=Math.ceil(Math.log10(lo)); e<=Math.floor(Math.log10(hi)); e++){
    const v=Math.pow(10,e);
    g.appendChild(el("line",{x1:0,x2:iw,y1:sy(v),y2:sy(v),stroke:"var(--grid)","stroke-width":1}));
    const t=el("text",{x:-9,y:sy(v)+4,"text-anchor":"end",fill:"var(--muted)","font-size":11});
    t.textContent=v>=1?v.toFixed(0):v.toFixed(1); g.appendChild(t);
  }
  betas.forEach((b,i)=>{const t=el("text",{x:sx(i),y:ih+20,"text-anchor":"middle",
    fill:"var(--muted)","font-size":11}); t.textContent=b; g.appendChild(t);});
  const xl=el("text",{x:iw/2,y:ih+40,"text-anchor":"middle",fill:"var(--muted)","font-size":11});
  xl.textContent="haze β (1/m) — extinction coefficient"; g.appendChild(xl);
  const yl=el("text",{transform:`translate(-44,${ih/2}) rotate(-90)`,"text-anchor":"middle",
    fill:"var(--muted)","font-size":11}); yl.textContent="ATE RMSE (cm, log)"; g.appendChild(yl);

  // do-nothing 바닥선은 **시퀀스마다** 그린다. 하나만 그리면 다른 시퀀스의
  // 곡선을 남의 바닥값에 대고 읽게 된다 - fr1 desk 의 바닥은 57.6 cm 로
  // fr1 xyz 의 17.2 cm 와 3 배 넘게 다르다. 선 하나가 그 오독을 만든다.
  D.sequences.forEach((s,si)=>{
    const floor=s.rows.length?s.rows[0].identity:null;
    if(!floor) return;
    g.appendChild(el("line",{x1:0,x2:iw,y1:sy(floor),y2:sy(floor),
      stroke:"var(--muted)","stroke-width":1.5,
      "stroke-dasharray": si?"2 3":"5 4"}));
    const t=el("text",{x:iw-6,y:sy(floor)-6,"text-anchor":"end",fill:"var(--muted)",
      "font-size":10.5});
    t.textContent=`do-nothing floor · ${shortName(s.name)} (${floor.toFixed(0)} cm)`;
    g.appendChild(t);
  });

  D.sequences.forEach((s,si)=>{
    ["baseline","wme"].forEach(k=>{
      const pts=[];
      s.rows.forEach(r=>{const v=r.runs&&r.runs[k];
        const i=betas.indexOf(r.beta); if(v&&i>=0) pts.push([i,v.ate]);});
      if(pts.length<2) return;
      const d=pts.map((p,i)=>(i?"L":"M")+sx(p[0]).toFixed(1)+" "+sy(p[1]).toFixed(1)).join(" ");
      g.appendChild(el("path",{d,fill:"none",stroke:SY[k].c,"stroke-width":2,
        "stroke-linejoin":"round",
        ...(si?{"stroke-dasharray":"6 4"}:{})}));
      pts.forEach(p=>{
        const c=el("circle",{cx:sx(p[0]),cy:sy(p[1]),r:4,fill:SY[k].c,
          stroke:"var(--surface)","stroke-width":2});
        c.addEventListener("mousemove",e=>showTip(e,
          `<b>${s.name}</b> · β=${betas[p[0]]}<br>${SY[k].label}<br>ATE <b>${fmt(p[1])} cm</b>`));
        c.addEventListener("mouseleave",hideTip);
        g.appendChild(c);
      });
    });
  });
  g.appendChild(el("line",{x1:0,x2:iw,y1:ih,y2:ih,stroke:"var(--axis)","stroke-width":1}));
  document.getElementById("degplot").appendChild(svg);

  // 실패 양상. 이게 없으면 위 그림은 고전 파이프라인에게 유리하게 오독된다.
  const last=D.sequences[0].rows[D.sequences[0].rows.length-1];
  const bl=last.runs.baseline, wm=last.runs.wme;
  if(bl&&wm) document.getElementById("degnote").innerHTML =
    `<strong>The classical line flattening at high β is not robustness — it is surrender.</strong>
     At β=${last.beta} it has lost tracking on <b>${(100*bl.lost/bl.frames).toFixed(0)}%</b> of
     frames and holds the last pose, so its trajectory is nearly a frozen camera and it scores
     ${fmt(bl.ate/last.identity)}× the do-nothing floor — because that is what it has become.
     WME loses tracking on <b>${(100*wm.lost/wm.frames).toFixed(0)}%</b> of frames and drifts to
     ${fmt(wm.ate/last.identity)}× the floor. Neither degrades gracefully; one fails loudly and
     stops, the other fails silently and keeps answering.`;
})();

/* ---------- loop closure ---------- */
(function(){
  const L = DATA.loop;
  if(!L || !L.modes || !Object.keys(L.modes).length) return;
  document.getElementById("loop-sec").hidden = false;
  let h=`<table><thead><tr><th>place recognition</th><th>odometry ATE</th>
    <th>+ loop closure</th><th>change</th><th>loops accepted</th>
    <th>loop-edge error</th><th>wrong place</th></tr></thead><tbody>`;
  for(const k of ["orb","tcg"]){
    const m=L.modes[k]; if(!m) continue;
    const ch=(m.before-m.after)/m.before*100;
    h+=`<tr><td>${m.label}</td><td>${fmt(m.before)} cm</td>
        <td>${fmt(m.after)} cm</td>
        <td class="${ch>0?'win':'lose'}">${ch>0?'−':'+'}${Math.abs(ch).toFixed(1)}%</td>
        <td>${m.n}</td>
        <td>${m.trans_median_cm!=null?fmt(m.trans_median_cm)+' cm':'—'}</td>
        <td class="${m.gross_false>m.scored/2?'lose':''}">${m.gross_false}/${m.scored}</td></tr>`;
  }
  h+=`</tbody></table>`;
  document.getElementById("loop").innerHTML = h +
    `<p class="sub" style="margin-top:12px"><strong>Read the last two columns first.</strong>
     A loop count without loop accuracy cannot distinguish accepting many right loops from
     accepting a few wrong ones. Descriptor loop closure recovers 57% of the drift from 44 edges
     accurate to ~2 cm. The constellation tier accepts 3 and recovers 31% from them, at ~15 cm
     edge accuracy — far lower recall, usable precision. Only 25 of 92 keyframes contained the
     4+ objects a constellation query needs at all; an office is closer to an empty corridor
     than the architecture assumed. Net, the WME full system reaches 15.06 cm against the
     classical 20.67 cm — but that lead comes from the front-end, not the loop closure.</p>`;
})();

/* ---------- sequence selector ---------- */
const sel=document.getElementById("seq");
DATA.sequences.forEach((s,i)=>{const o=document.createElement("option");
  o.value=i; o.textContent=s.name; sel.appendChild(o);});
function render(){
  const s=DATA.sequences[+sel.value];
  const p=document.getElementById("panels"); p.innerHTML="";
  const L=document.createElement("div"), R=document.createElement("div");
  p.appendChild(L); p.appendChild(R);
  drawPanel(L,s,"baseline");
  const bw=bestWme(s);
  drawPanel(R,s,bw?bw.k:"wme");
  drawSeries(s);
  legend("lgd2",["baseline",bw?bw.k:"wme"],
    `<span class="lg" style="color:var(--muted)">faint = per frame · solid = 9-frame rolling median</span>`);
}
sel.addEventListener("change",render);
sel.value=DATA.sequences.findIndex(s=>s.name.includes("walking_xyz"));
if(+sel.value<0) sel.value=0;
render();

document.getElementById("foot").innerHTML=`
  <p style="color:var(--muted)"><strong>Method.</strong> ${DATA.sequences.length} TUM RGB-D
  sequences. Both systems: identical intrinsics, distortion coefficients, depth scale (5000),
  rgb↔depth association (20 ms) and keyframe rule (0.03 m). Estimation in C++, scoring in
  Python — deliberately split so one codebase does not grade its own output. ATE is
  Umeyama-aligned RMSE; RPE is frame-to-frame relative translation error. Timing is
  end-to-end wall clock including image I/O, CPU only.</p>
  <p style="color:var(--muted)"><strong>Limits.</strong> Odometry only — neither system has
  loop closure or bundle adjustment, so drift is unbounded by construction. The baseline is a
  self-implemented ORB front-end, not ORB-SLAM3, and a self-implemented control can always be
  suspected of being under-tuned: it is reported here at ORB-SLAM3's published front-end
  settings, tracking ~1000 keypoints with ~455 PnP inliers per frame and zero tracking losses
  on the static sequences. One dataset family, short windows. No claim of
  state-of-the-art is made or implied.</p>`;
</script>
"""


def load_extras() -> dict:
    """열화 스윕과 루프 클로저 결과. 없으면 그 절을 통째로 감춘다 -
    빈 표를 그리면 '측정했는데 0' 처럼 보인다."""
    out = {}
    dp = ROOT / "results" / "degrade" / "degrade.json"
    if dp.exists():
        out["degrade"] = json.loads(dp.read_text(encoding="utf-8"))
    lp = ROOT / "results" / "loop" / "loop.json"
    if lp.exists():
        out["loop"] = json.loads(lp.read_text(encoding="utf-8"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="results/bench/benchmark.json")
    ap.add_argument("--out", dest="dst", default="results/bench/index.html")
    a = ap.parse_args()

    src = ROOT / a.src
    report = json.loads(src.read_text(encoding="utf-8"))
    data = compact(report)
    data.update(load_extras())
    html = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    dst = ROOT / a.dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(html, encoding="utf-8")
    print(f"저장: {dst}  ({dst.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
