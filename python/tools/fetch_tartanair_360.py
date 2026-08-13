# TartanAir V2 의 360 등장방형 시퀀스 한 토막을 이 저장소의 TUM 배치로 만든다.
#
# 이 저장소에는 360 주행 자료가 없었다. 그래서 `tools/equirect_convert.cpp` 가
# 합성 검증만 받은 채로 남아 있었고, 뷰어의 시퀀스 목록에도 360 항목이 없었다.
# 이 스크립트가 그 자리를 메운다 - 받아서, 원근으로 잘라, 배치로 쓰고, 매니페스트
# 한 줄까지 찍는다.
#
# **왜 전체를 안 받는가.** 등장방형 zip 하나가 실내 1.1 GB, 야외 3.1 GB 다.
# 우리가 쓰는 것은 수십 프레임이다. HuggingFace 가 `Accept-Ranges: bytes` 를
# 주고 206 으로 답하므로, zip 끝의 중앙 디렉터리만 읽어 목록을 얻고 필요한
# 멤버의 구간만 다시 요청하면 된다. 50 프레임 스테레오 기준 실측 약 495 MB 다.
# 서버가 범위 요청을 거절하면 조용히 전체를 받는 대신 **실패한다** - 수 GB 를
# 모르는 사이에 받는 것이 이 스크립트에서 가장 나쁜 실패다.
#
# **왜 equirect_convert 로 깊이를 만들지 않는가.** 두 가지가 겹친다.
#   1. 이 리그의 베이스라인은 파노라마 방위 90 도에 놓여 있다. 그 도구의
#      `checkRectified` 는 기본 가정이 방위 0 도라 판정이 거꾸로 나온다
#      (그래서 `--baseline-yaw` 를 붙였다). 실측: 쓸 수 있는 뷰가 yaw 90/270.
#   2. 그걸 고쳐도 이 장면은 최근접이 0.125 m 다. B = 0.25 m 에서 시차가
#      fx*B/Z = 320*0.25/0.125 = 640 화소, 곧 영상 폭 전체라 정합이 성립하지
#      않는다.
# 그래서 추정 대신 **데이터셋이 함께 주는 렌더 GT 깊이** 를 옮긴다. 옮기는
# 방식은 `python/wme/reference/equirect.py` 의 것을 그대로 쓴다 - 그 파일이
# equirect_convert 의 numpy 오라클이므로 규약이 어긋날 자리가 없다.
#
# **이 변환이 맞다는 근거 (실측).** TartanAir 는 같은 프레임의 정면 핀홀
# 깊이도 따로 배포한다. 등장방형 깊이를 여기 방식으로 옮긴 결과와 그 정면
# 깊이의 차이가 **중앙값 0.0016 m** 였다. 경도 규약 · 90 도 회전 보정 ·
# 반경거리→광축거리 변환 · 바이트 순서가 **모두** 맞아야 나오는 값이다.
#
# **정직하게 남길 한계.** TartanAir 의 모션은 드론/보행 궤적이지 차량 주행이
# 아니다. 뷰어는 도로 장면을 전제하므로 실내 방을 "건물 몇 동" 으로 라벨한다.
# 그리고 90 도 한 방향만 잘라내므로 구면 전체를 쓰는 것이 아니다.
#
# 쓰는 법:
#   python python/tools/fetch_tartanair_360.py --out data/tartanair_360 --frames 50
#   (--env / --difficulty / --traj 로 다른 구간을 고를 수 있다)

from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
import zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wme.reference.equirect import WRAP_PAD, build_maps, derive_intrinsics

BASE = "https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/"

# PNG 값 = m * DEPTH_SCALE. uint16 상한이 65535 이므로 표현 한계가 65.5 m 다.
# 실내 장면에서 그 너머는 창밖 하늘이고 어차피 유효 깊이로 쓰지 않는다.
DEPTH_SCALE = 1000.0


class HttpFile(io.RawIOBase):
    """HTTP 범위 요청으로 seek 가능한 파일처럼 보이게 한다.

    `nread` 에 실제로 받은 바이트를 쌓아 둔다 - 예산을 지켰는지 사후에 물어볼
    수 있어야 한다.
    """

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        self.nread = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            if r.headers.get("Accept-Ranges") != "bytes":
                raise RuntimeError(
                    "서버가 범위 요청을 지원하지 않는다. 전체를 받게 되므로 "
                    "멈춘다: " + url)

    def seek(self, off, whence=0):
        self.pos = (off if whence == 0
                    else self.pos + off if whence == 1
                    else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0:
            return b""
        end = min(self.pos + n, self.size) - 1
        if end < self.pos:
            return b""
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=180) as r:
            if r.status != 206:
                raise RuntimeError(
                    f"206 이 아니라 {r.status} 가 왔다 - 범위 요청이 무시됐다: "
                    f"{self.url}")
            data = r.read()
        self.pos += len(data)
        self.nread += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        memoryview(b).cast("B")[:len(d)] = d
        return len(d)


def open_zip(rel: str):
    f = HttpFile(BASE + rel)
    return zipfile.ZipFile(io.BufferedReader(f, 256 * 1024)), f


def decode_depth(buf: bytes) -> np.ndarray:
    """TartanAir 깊이 PNG -> float32 (m).

    float32 를 H x W x 4 8 비트로 무손실 포장한 것이라 바이트를 그대로 다시
    읽는다. 순서를 뒤집으면 정면 정답과 중앙 0.95 m 어긋난다 (실측).
    """
    a = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    r = np.ascontiguousarray(a)
    return np.frombuffer(r.tobytes(), dtype=np.float32).reshape(
        a.shape[0], a.shape[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--env", default="ArchVizTinyHouseDay")
    ap.add_argument("--difficulty", default="Data_easy")
    ap.add_argument("--traj", default="P000")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="시각 합성용. TartanAir V2 는 10 Hz 다")
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=640)
    a = ap.parse_args()

    base = f"{a.env}/{a.difficulty}"
    os.makedirs(f"{a.out}/rgb", exist_ok=True)
    os.makedirs(f"{a.out}/depth", exist_ok=True)

    zr, fr = open_zip(f"{base}/image_lcam_equirect.zip")
    zd, fd = open_zip(f"{base}/depth_lcam_equirect.zip")

    # 포즈와 시각은 영상 zip 안에 함께 들어 있다.
    def member(zf, suffix):
        hit = [n for n in zf.namelist() if n.endswith(suffix) and a.traj in n]
        if not hit:
            raise SystemExit(f"{suffix} 를 zip 에서 찾지 못했다 ({a.traj})")
        return hit[0]

    poses = zr.read(member(zr, "pose_lcam_front.txt")).decode().strip().splitlines()
    # **시각은 합성값이다.** 이 zip 에는 촬영 시각 파일이 없다 (멤버를 전수
    # 확인했다 - 포즈 txt 와 영상뿐이다). TartanAir V2 가 10 Hz 라고 밝히고
    # 있으므로 idx/10 으로 만든다. 실제 촬영 간격이 아니라 규칙적인 격자다.
    # 이 배치를 시간 정합에 쓰는 곳이 생기면 그 사실을 알고 써야 한다.
    stamps = [i / a.fps for i in range(len(poses))]
    n = min(a.frames, len(poses), len(stamps))
    if n < a.frames:
        print(f"# 요청 {a.frames} 보다 적은 {n} 프레임만 있다")

    view = derive_intrinsics(a.width, a.height, a.hfov, 0.0, 0.0)
    mx, my = build_maps(view, 2048, 1024)
    uu, vv = np.meshgrid(np.arange(float(a.width)), np.arange(float(a.height)))
    xr = (uu - view.cx) / view.fx
    yr = (vv - view.cy) / view.fy
    # 등장방형 깊이는 광선을 따라 잰 **반경 거리** 다. 핀홀 깊이는 광축 성분
    # 이므로 cos 를 곱해 옮긴다. 이 항이 빠지면 화면 가장자리가 멀어진다.
    cos_t = 1.0 / np.sqrt(xr * xr + yr * yr + 1.0)

    def warp(img, interp):
        pad = np.concatenate(
            [img[:, -WRAP_PAD:], img, img[:, :WRAP_PAD]], axis=1)
        return cv2.remap(pad, mx + WRAP_PAD, my, interp)

    rgb_idx, dep_idx, lo_all, hi_all = [], [], [], []
    for i in range(n):
        # **90 도 굴린다.** 이 리그의 등장방형은 중앙이 정면이 아니다 - yaw 90
        # 잘라내기가 데이터셋 자신의 lcam_front 와 화소 단위로 일치하는 것을
        # 확인했다. 굴려 두면 yaw 0 이 정면이 되어 pose_lcam_front 가 그대로
        # 이 뷰의 포즈가 된다.
        eq_rgb = cv2.imdecode(
            np.frombuffer(zr.read(
                f"{base}/{a.traj}/image_lcam_equirect/"
                f"{i:06d}_lcam_equirect_image.png"), np.uint8), cv2.IMREAD_COLOR)
        eq_rgb = np.roll(eq_rgb, -eq_rgb.shape[1] // 4, axis=1)

        eq_dep = decode_depth(zd.read(
            f"{base}/{a.traj}/depth_lcam_equirect/"
            f"{i:06d}_lcam_equirect_depth.png"))
        eq_dep = np.roll(eq_dep, -eq_dep.shape[1] // 4, axis=1)

        rgb = warp(eq_rgb, cv2.INTER_CUBIC)
        # 깊이는 절대 섞지 않는다. 앞뒤 경계에서 평균은 어느 물체도 아닌 값이다.
        dep = warp(eq_dep, cv2.INTER_NEAREST) * cos_t

        valid = np.isfinite(dep) & (dep > 0)
        if valid.any():
            lo_all.append(np.percentile(dep[valid], 1))
            hi_all.append(np.percentile(dep[valid], 99))
        q = np.zeros(dep.shape, np.uint16)
        ok = valid & (dep * DEPTH_SCALE < 65535)
        q[ok] = (dep[ok] * DEPTH_SCALE).astype(np.uint16)   # 나머지는 0 = 결측

        cv2.imwrite(f"{a.out}/rgb/{i:06d}.png", rgb)
        cv2.imwrite(f"{a.out}/depth/{i:06d}.png", q)
        rgb_idx.append(f"{stamps[i]:.6f} rgb/{i:06d}.png")
        dep_idx.append(f"{stamps[i]:.6f} depth/{i:06d}.png")
        if i % 10 == 0:
            print(f"  {i}/{n}  누적 "
                  f"{(fr.nread + fd.nread) / 1024 / 1024:.0f} MB", flush=True)

    with open(f"{a.out}/rgb.txt", "w") as f:
        f.write("# timestamp filename\n" + "\n".join(rgb_idx) + "\n")
    with open(f"{a.out}/depth.txt", "w") as f:
        f.write("# timestamp filename\n" + "\n".join(dep_idx) + "\n")

    # TartanAir 포즈는 `tx ty tz qx qy qz qw` (NED, 정면 카메라) 라 컬럼 순서가
    # TUM 과 같다. 시각만 앞에 붙인다. NED 와 카메라계는 축 순환 치환(회전)이라
    # bench_viewer 의 Kabsch 정렬이 흡수한다 - 반사가 아님은 확인했다. 여기서
    # 임의로 축을 바꾸면 그 자리가 조용히 틀리는 자리가 된다.
    with open(f"{a.out}/groundtruth.txt", "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for t, p in zip(stamps[:n], poses[:n]):
            f.write(f"{t:.7f} {' '.join(p.split())}\n")

    lo, hi = float(np.min(lo_all)), float(np.max(hi_all))
    with open(f"{a.out}/calib.txt", "w") as f:
        f.write(
            f"# wme: TartanAir V2 {base}/{a.traj} 등장방형 -> 정면 "
            f"{a.hfov:.0f} 도 원근 뷰.\n"
            "# 깊이는 데이터셋의 **렌더 GT** 이지 추정값이 아니다.\n"
            "# 내부파라미터는 hfov 와 출력 크기에서 유도한 값이다.\n"
            f"fx: {view.fx}\nfy: {view.fy}\n"
            f"cx: {view.cx}\ncy: {view.cy}\n"
            f"width: {view.width}\nheight: {view.height}\n"
            f"depth_scale: {DEPTH_SCALE}\n"
            f"# depth_min/max 는 이 {n} 프레임에서 실측한 p1 의 최소, p99 의 최대다.\n"
            f"depth_min: {lo:.3f}\ndepth_max: {hi:.3f}\n"
            "dist: 0 0 0 0 0\n")

    mb = (fr.nread + fd.nread) / 1024 / 1024
    print(f"\n# 범위 요청으로 받은 바이트: {mb:.1f} MB "
          f"(zip 전체는 {(fr.size + fd.size) / 1e9:.2f} GB)")
    print(f"# 실측 깊이 범위: p1 최소 {lo:.3f} m, p99 최대 {hi:.3f} m")
    print(f"완료: {a.out}  ({n} 프레임)\n")
    name = f"tartanair_{a.env.lower()}_360"
    print("# results/bench/viewer.tsv 에 넣을 줄 (탭 구분):")
    print(f"SEQ\t{name}\ttartanair\t{os.path.abspath(a.out)}\t"
          f"{os.path.abspath(a.out)}/groundtruth.txt\t0")
    print("# 뷰어는 기본으로 kitti 만 싣는다 - 이 시퀀스를 보려면 --all 1 이 필요하다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
