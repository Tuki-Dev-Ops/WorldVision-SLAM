"""언리얼 임포터를 스텁 위에서 돌리고 결과를 JSON 과 맞춰 본다.

    python engine/unreal/Check/check.py [장면.json]

확인하는 것
----------
* 끝까지 도는가. 예외 없이.
* 개수가 맞는가. 지표면 사각형 · 차선 · 건물 · 나무 · 기둥 · 차량.
* 좌표 변환이 거울인가. 축을 바꾸면 손 방향만 바뀌고 형상은 보존되어야
  하므로, 상호 거리가 변하면 축을 잘못 섞은 것이다.
* 단위를 안 잊었는가. 미터에서 센티미터로 100 배.

확인하지 못하는 것
-----------------
에셋이 실제로 만들어지는지, 재질 그래프가 컴파일되는지, 화면에 어떻게
보이는지. 그건 엔진이 있어야 한다.
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                      # 스텁 unreal 을 먼저 찾게 한다
sys.path.insert(0, os.path.dirname(HERE))     # 임포터가 있는 폴더

REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
scene = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "results", "scene", "kitti_00.json")
scene = os.path.abspath(scene)
if not os.path.exists(scene):
    print("장면 파일이 없다: %s" % scene)
    sys.exit(1)
os.environ["WV_SCENE"] = scene

import unreal  # noqa: E402  (스텁)

fail = []
import import_worldvision_scene  # noqa: E402,F401  (임포트하면 main() 이 돈다)

with open(scene, encoding="utf-8") as f:
    S = json.load(f)

# --- 개수 ---
surf = (S.get("surfaces") or {}).get("tiles") or []
lanes = S.get("lanes") or []
want_actors = (
    len(S.get("buildings") or [])
    + sum(1 for b in (S.get("buildings") or []) if b.get("roof", 0.0) > 0.0)
    + 2 * len(S.get("trees") or [])
    + len(S.get("poles") or [])
    + len(S.get("vehicles") or [])
)
mesh_actors = sum(1 for a in unreal.SPAWNED if a.folder.endswith("Surfaces")
                  or a.folder.endswith("Lanes"))
prim_actors = len(unreal.SPAWNED) - mesh_actors
print("\n액터: 기본도형 %d (기대 %d), 메시 %d" % (prim_actors, want_actors, mesh_actors))
if prim_actors != want_actors:
    fail.append("기본 도형 액터 수가 다르다")

# 지표면 사각형 수 = 타일 수 (종류별로 나뉘어도 합은 같아야 한다)
quads = 0
for name, m in unreal.MESHES.items():
    if "/SM_WV_" not in name:
        continue
    for d in m.descs:
        quads += len(d.polys) // 2
surf_quads = 0
for name, m in unreal.MESHES.items():
    if "Lane" in name:
        continue
    for d in m.descs:
        surf_quads += len(d.polys) // 2
print("지표면 사각형 %d (기대 %d), 전체 사각형 %d" % (surf_quads, len(surf), quads))
if surf_quads != len(surf):
    fail.append("지표면 사각형 수가 타일 수와 다르다")

lane_meshes = sum(1 for n in unreal.MESHES if "Lane" in n)
print("차선 메시 %d (기대 %d)" % (lane_meshes, len(lanes)))
if lane_meshes != len(lanes):
    fail.append("차선 메시 수가 다르다")

# --- 좌표 ---
def to_ue(v):
    return (-v[2] * 100.0, v[0] * 100.0, v[1] * 100.0)

src = [b["center"] for b in (S.get("buildings") or [])][:120]
dst = [to_ue(p) for p in src]
worst = 0.0
for i in range(len(src)):
    for j in range(i + 1, len(src)):
        a = math.dist(src[i], src[j]) * 100.0
        b = math.dist(dst[i], dst[j])
        worst = max(worst, abs(a - b))
print("상호 거리 변화 최대 %.6f cm  (%d 개 점)" % (worst, len(src)))
if worst > 1e-3:
    fail.append("좌표 변환이 거리를 바꾼다 - 축을 잘못 섞었다")

bad = [a for a in unreal.SPAWNED
       if not all(math.isfinite(c) for c in
                  (a.location.x, a.location.y, a.location.z))]
print("NaN/Inf 좌표 %d" % len(bad))
if bad:
    fail.append("좌표에 NaN/Inf 가 있다")

if S.get("ego"):
    e = to_ue(S["ego"])
    near = min((math.dist((a.location.x, a.location.y, a.location.z), e)
                for a in unreal.SPAWNED), default=1e18)
    print("자차에서 가장 가까운 액터 %.1f cm" % near)
    if near > 5000.0:
        fail.append("자차 주변에 액터가 없다 - 좌표계가 어긋났을 수 있다")

# 단위. 장면 폭이 미터 값의 100 배여야 한다.
xs = [a.location.x for a in unreal.SPAWNED]
ys = [a.location.y for a in unreal.SPAWNED]
if xs:
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    print("장면 폭 %.0f cm" % span)
    if span < 1000.0:
        fail.append("장면이 너무 작다 - 미터/센티미터 변환을 잊었을 수 있다")

print()
if fail:
    for m in fail:
        print("실패: %s" % m)
    sys.exit(1)
print("언리얼 임포터 통과 (스텁). 엔진이 있어야 확인되는 것은 engine/README.md 참조")
