"""WorldVision-SLAM 장면을 언리얼 레벨에 세운다.

GLB 를 임포트해도 보이기는 하지만 그것은 굳은 삼각형 덩이다. 집 하나가 벽이
아니라 커다란 메시의 일부라 충돌체도 재질도 따로 줄 수 없고, 차를 움직이거나
건물을 지울 수도 없다.

이 스크립트는 --export-json 이 낸 매개변수 목록을 읽어 액터를 놓는다. 그때
부터는 언리얼이 아는 StaticMeshActor 라, 블루프린트로 바꾸든 물리를 붙이든
그다음은 엔진의 일이 된다.

사용
----
    에디터에서 Output Log 를 열고 (Window > Output Log), Cmd 입력줄에서:

        py "D:/Program Files/Slam-Model/engine/unreal/import_worldvision_scene.py"

    배치 모드로는:

        UnrealEditor-Cmd.exe <프로젝트.uproject> -run=pythonscript
            -script="D:/.../import_worldvision_scene.py"

    기본 경로는 아래 SCENE_JSON 이다. 다른 파일을 쓰려면 그 값을 바꾸거나
    환경변수 WV_SCENE 에 경로를 넣는다.

무엇이 들어오는가
----------------
    지표면   도로 · 인도 · 잔디 · 기타. 종류마다 메시와 재질이 따로다 -
             차가 굴러가는 아스팔트와 밟고 서는 보도가 같은 물성일 수 없다.
    노면무늬 0.1 m 로 구운 밝기 PNG. 차선은 높이 차이가 0 이라 기하로는
             나오지 않으므로 무늬로 온다.
    차선     도로 구조에서 **예측한** 선. 관측한 무늬와 섞지 않는다 -
             측정과 모형은 따로 두어야 서로를 채점할 수 있다.
    건물 · 나무 · 기둥 · 차량

좌표에 대하여
------------
내보낸 좌표는 오른손계에 +y 가 위이고 단위는 미터다 (glTF 규약). 언리얼은
왼손계에 **+z 가 위** 이고 단위는 센티미터다. 그래서 축을 바꾸고 100 을
곱한다 - 축만 바꾸고 단위를 잊으면 장면이 100 배 작게 들어와 원점의 점처럼
보인다.

    UE.X = -gltf.z      (앞)
    UE.Y =  gltf.x      (오른쪽)
    UE.Z =  gltf.y      (위)
"""

import json
import math
import os

import unreal

SCENE_JSON = os.environ.get(
    "WV_SCENE",
    r"D:/Program Files/Slam-Model/results/scene/kitti_00.json")

M = 100.0  # 미터 -> 센티미터

PKG = "/Game/WorldVision"

# 엔진 기본 도형. 크기가 100 cm 인 정육면체/구/원기둥이라 스케일 1 이 1 m 다.
CUBE = "/Engine/BasicShapes/Cube.Cube"
SPHERE = "/Engine/BasicShapes/Sphere.Sphere"
CYLINDER = "/Engine/BasicShapes/Cylinder.Cylinder"


def to_ue(v):
    """오른손 +y up (m) -> 언리얼 왼손 +z up (cm)."""
    return unreal.Vector(-v[2] * M, v[0] * M, v[1] * M)


def dir_ue(v):
    """방향 벡터. 원점 보정이 없을 뿐 축 변환은 같고, 단위는 붙이지 않는다."""
    return unreal.Vector(-v[2], v[0], v[1])


def yaw_from(fwd):
    """진행 방향에서 요각을 뽑는다. 지면 위 회전만 쓰므로 yaw 하나면 된다."""
    x = -fwd[2]
    y = fwd[0]
    if abs(x) < 1e-6 and abs(y) < 1e-6:
        return unreal.Rotator(0.0, 0.0, 0.0)
    return unreal.Rotator(0.0, 0.0, math.degrees(math.atan2(y, x)))


def spawn(mesh_path, loc, rot, scale, label, folder):
    mesh = unreal.EditorAssetLibrary.load_asset(mesh_path)
    if mesh is None:
        unreal.log_error("메시를 못 읽었다: %s" % mesh_path)
        return None
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.StaticMeshActor, loc, rot)
    actor.static_mesh_component.set_static_mesh(mesh)
    actor.set_actor_scale3d(scale)
    actor.set_actor_label(label)
    # 폴더로 묶어 두면 아웃라이너에서 클래스별로 다룰 수 있다.
    actor.set_folder_path("WorldVision/%s" % folder)
    return actor


# ---------------------------------------------------------------------------
# 재질 — 그래프를 손으로 엮는다
# ---------------------------------------------------------------------------
# 파이썬에서 재질을 만들려면 노드를 직접 연결해야 한다. 번거롭지만 프로젝트에
# 무엇이 들어 있는지 가정하지 않아도 되는 유일한 길이다 - 부모 재질을 찍어
# 두면 그것이 없는 프로젝트에서 통째로 분홍색이 된다.
def make_material(name, rgb, rough=0.85, texture=None):
    path = "%s/%s" % (PKG, name)
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.EditorAssetLibrary.load_asset(path)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    mat = tools.create_asset(name, PKG, unreal.Material,
                             unreal.MaterialFactoryNew())
    if mat is None:
        return None
    lib = unreal.MaterialEditingLibrary
    if texture is not None:
        node = lib.create_material_expression(
            mat, unreal.MaterialExpressionTextureSample, -400, 0)
        node.texture = texture
        lib.connect_material_property(node, "RGB",
                                      unreal.MaterialProperty.MP_BASE_COLOR)
    else:
        node = lib.create_material_expression(
            mat, unreal.MaterialExpressionConstant3Vector, -400, 0)
        node.constant = unreal.LinearColor(rgb[0], rgb[1], rgb[2], 1.0)
        lib.connect_material_property(node, "",
                                      unreal.MaterialProperty.MP_BASE_COLOR)
    r = lib.create_material_expression(
        mat, unreal.MaterialExpressionConstant, -400, 220)
    r.r = rough
    lib.connect_material_property(r, "", unreal.MaterialProperty.MP_ROUGHNESS)
    lib.recompile_material(mat)
    unreal.EditorAssetLibrary.save_asset(path)
    return mat


def import_texture(png_path, name):
    """구워 온 노면 밝기 PNG 를 텍스처 에셋으로 들인다."""
    path = "%s/%s" % (PKG, name)
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.EditorAssetLibrary.load_asset(path)
    if not os.path.exists(png_path):
        unreal.log_warning("노면 지도가 없다: %s" % png_path)
        return None
    task = unreal.AssetImportTask()
    task.filename = png_path
    task.destination_path = PKG
    task.destination_name = name
    task.automated = True
    task.replace_existing = True
    task.save = True
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    tex = unreal.EditorAssetLibrary.load_asset(path)
    if tex is not None:
        # 길 밖으로 새면 노면이 반복되어 없는 도로가 생긴다.
        tex.address_x = unreal.TextureAddress.TA_CLAMP
        tex.address_y = unreal.TextureAddress.TA_CLAMP
    return tex


# ---------------------------------------------------------------------------
# 메시 — 사각형 수천 장을 액터 하나로
# ---------------------------------------------------------------------------
# 타일마다 액터를 두면 1 만 2 천 개가 된다. 아웃라이너가 마비되고 드로콜이
# 그만큼 나가는데, 어차피 이어진 한 장의 바닥이라 나눌 이유가 없다.
def build_mesh(name, quads, material, folder, label):
    """quads: [(p0, p1, p2, p3, uv0..uv3)] - 각 점은 언리얼 좌표(cm)."""
    if not quads:
        return None
    path = "%s/%s" % (PKG, name)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.EditorAssetLibrary.delete_asset(path)
    sm = tools.create_asset(name, PKG, unreal.StaticMesh,
                            unreal.StaticMeshFactoryNew())
    if sm is None:
        unreal.log_error("스태틱 메시를 못 만들었다: %s" % path)
        return None
    md = sm.create_static_mesh_description()
    pg = md.create_polygon_group()
    for q in quads:
        pts, uvs = q[0], q[1]
        vids = []
        for p in pts:
            vid = md.create_vertex()
            md.set_vertex_position(vid, p)
            vids.append(vid)
        vis = []
        for k, vid in enumerate(vids):
            vi = md.create_vertex_instance(vid)
            md.set_vertex_instance_uv(vi, unreal.Vector2D(uvs[k][0], uvs[k][1]), 0)
            vis.append(vi)
        # 사각형 하나를 삼각형 둘로. 언리얼은 시계방향이 앞면이다.
        md.create_polygon(pg, [vis[0], vis[1], vis[2]])
        md.create_polygon(pg, [vis[0], vis[2], vis[3]])
    sm.build_from_static_mesh_descriptions([md])
    if material is not None:
        sm.set_editor_property("static_materials",
                               [unreal.StaticMaterial(material_interface=material)])
    unreal.EditorAssetLibrary.save_asset(path)

    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.StaticMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    actor.static_mesh_component.set_static_mesh(sm)
    actor.set_actor_label(label)
    actor.set_folder_path("WorldVision/%s" % folder)
    return actor


SURF = [
    ("Road",     "M_WV_Asphalt",  (0.16, 0.16, 0.17), 0.90),
    ("Sidewalk", "M_WV_Sidewalk", (0.42, 0.41, 0.38), 0.80),
    ("Grass",    "M_WV_Grass",    (0.07, 0.16, 0.05), 0.95),
    ("Ground",   "M_WV_Ground",   (0.24, 0.23, 0.22), 0.90),
]


def build_surfaces(s, json_dir):
    surf = s.get("surfaces") or {}
    tiles = surf.get("tiles") or []
    if not tiles:
        return 0
    cell = float(surf.get("cell", 0.5))
    h = cell * 0.5 * M

    rm = s.get("road_map") or {}
    ok = bool(rm) and rm.get("width", 0) > 0
    if ok:
        org = to_ue(rm["origin"])
        au = dir_ue(rm["axis_u"])
        av = dir_ue(rm["axis_v"])
        span_u = max(1e-3, rm["width"] * rm["cell"]) * M
        span_v = max(1e-3, rm["height"] * rm["cell"]) * M

    tex = None
    if ok:
        tex = import_texture(os.path.join(json_dir, rm["image"]), "T_WV_Road")

    def uv(p):
        if not ok:
            return (0.5, 0.5)
        d = unreal.Vector(p.x - org.x, p.y - org.y, p.z - org.z)
        return (d.dot(au) / span_u, d.dot(av) / span_v)

    made = 0
    for cls, (nm, mat_name, rgb, rough) in enumerate(SURF):
        quads = []
        for t in tiles:
            if int(t[4]) != cls:
                continue
            c = to_ue(t[:3])
            # 언리얼에서 수평면은 XY 이고 높이가 Z 다.
            pts = [unreal.Vector(c.x - h, c.y - h, c.z),
                   unreal.Vector(c.x + h, c.y - h, c.z),
                   unreal.Vector(c.x + h, c.y + h, c.z),
                   unreal.Vector(c.x - h, c.y + h, c.z)]
            quads.append((pts, [uv(p) for p in pts]))
        if not quads:
            continue
        # 포장면에만 구워 온 무늬를 입힌다. 잔디의 밝기는 풀색이 아니라
        # 그때의 노출이라 입히면 안 된다.
        mat = make_material(mat_name, rgb, rough,
                            tex if (cls in (0, 1) and tex is not None) else None)
        if build_mesh("SM_WV_%s" % nm, quads, mat, "Surfaces", nm):
            made += len(quads)
    return made


def build_lanes(s):
    lanes = s.get("lanes") or []
    if not lanes:
        return 0
    m_edge = make_material("M_WV_LaneEdge", (0.80, 0.80, 0.78), 0.35)
    m_ctr = make_material("M_WV_LaneCenter", (0.86, 0.74, 0.16), 0.35)
    made = 0
    for idx, L in enumerate(lanes):
        pts = L.get("points") or []
        if len(pts) < 2:
            continue
        center = L.get("kind") == "center"
        hw = max(0.05, float(L.get("width", 0.12))) * 0.5 * M
        quads = []
        for i in range(len(pts) - 1):
            # 중앙선은 파선이다. 실선은 추월 금지라는 뜻이 되는데, 그것은
            # 관측한 것이 아니라 지어낸 규칙이다.
            if center and (i % 2) == 1:
                continue
            a = to_ue(pts[i])
            b = to_ue(pts[i + 1])
            dx, dy = b.x - a.x, b.y - a.y
            n = math.hypot(dx, dy)
            if n < 1e-3:
                continue
            nx, ny = -dy / n * hw, dx / n * hw
            # 노면보다 2 cm 띄운다. 같은 높이면 z-fighting 으로 깜빡인다.
            lift = 2.0
            quads.append((
                [unreal.Vector(a.x - nx, a.y - ny, a.z + lift),
                 unreal.Vector(a.x + nx, a.y + ny, a.z + lift),
                 unreal.Vector(b.x + nx, b.y + ny, b.z + lift),
                 unreal.Vector(b.x - nx, b.y - ny, b.z + lift)],
                [(0, 0), (1, 0), (1, 1), (0, 1)]))
        if not quads:
            continue
        name = "SM_WV_Lane%02d" % idx
        if build_mesh(name, quads, m_ctr if center else m_edge, "Lanes",
                      ("LaneCenter " if center else "LaneEdge ") + str(idx)):
            made += 1
    return made


def main():
    if not os.path.exists(SCENE_JSON):
        unreal.log_error("장면 파일이 없다: %s" % SCENE_JSON)
        return
    with open(SCENE_JSON, "r", encoding="utf-8") as f:
        s = json.load(f)
    if s.get("format") != "worldvision-scene/1":
        unreal.log_error("형식이 맞지 않는다: %s" % s.get("format"))
        return
    json_dir = os.path.dirname(os.path.abspath(SCENE_JSON))

    unreal.EditorAssetLibrary.make_directory(PKG)

    n = {"buildings": 0, "trees": 0, "poles": 0, "vehicles": 0}
    n_surf = build_surfaces(s, json_dir)
    n_lane = build_lanes(s)

    for b in s.get("buildings", []):
        rot = yaw_from(b["forward"])
        loc = to_ue(b["center"])
        # 스케일은 (길이, 폭, 높이) 를 언리얼 축 (X 앞, Y 옆, Z 위) 에 맞춘다.
        spawn(CUBE, loc, rot,
              unreal.Vector(b["length"], b["width"], b["height"]),
              "Building", "Buildings")
        n["buildings"] += 1
        rise = b.get("roof", 0.0)
        if rise > 0.0:
            # 박공은 정육면체를 45 도 눕혀 얹는다. 기본 도형만 쓰면 별도
            # 에셋 없이 어떤 프로젝트에서든 돈다.
            top = unreal.Vector(loc.x, loc.y,
                                loc.z + (b["height"] * 0.5 + rise * 0.5) * M)
            r2 = unreal.Rotator(45.0, 0.0, rot.yaw)
            spawn(CUBE, top, r2,
                  unreal.Vector(b["length"], rise * 1.42, rise * 1.42),
                  "Roof", "Buildings")

    for t in s.get("trees", []):
        h = t["height"]
        ry = min(max(h * 0.35, 0.30), 2.8)
        rr = max(t["canopy"], ry * 0.70)
        cz = min(h * 0.58, h - ry * 0.6)
        th = max(0.2, cz - ry * 0.55)
        foot = t["foot"]
        trunk_loc = to_ue([foot[0], foot[1] + th * 0.5, foot[2]])
        spawn(CYLINDER, trunk_loc, unreal.Rotator(0, 0, 0),
              unreal.Vector(rr * 0.28, rr * 0.28, th),
              "Trunk", "Trees")
        leaf_loc = to_ue([foot[0], foot[1] + cz, foot[2]])
        spawn(SPHERE, leaf_loc, unreal.Rotator(0, 0, 0),
              unreal.Vector(rr * 2.0, rr * 2.0, ry * 2.0),
              "Canopy", "Trees")
        n["trees"] += 1

    for p in s.get("poles", []):
        foot = p["foot"]
        h = p["height"]
        spawn(CYLINDER, to_ue([foot[0], foot[1] + h * 0.5, foot[2]]),
              unreal.Rotator(0, 0, 0), unreal.Vector(0.12, 0.12, h),
              "Pole", "Poles")
        n["poles"] += 1

    for v in s.get("vehicles", []):
        size = v["size"]
        length = max(0.6, max(size))
        width = max(0.4, min(size[0], size[2]))
        height = max(0.4, size[1])
        label = "%s%s" % (v.get("class", "vehicle"),
                          " (moving)" if v.get("moving") else "")
        spawn(CUBE, to_ue(v["center"]), yaw_from(v["forward"]),
              unreal.Vector(length, width, height), label, "Vehicles")
        n["vehicles"] += 1

    unreal.log(
        "WorldVision: %s 프레임 %s - 지표면 %d 사각형, 차선 %d, "
        "건물 %d, 나무 %d, 기둥 %d, 차량 %d"
        % (s.get("sequence"), s.get("frame"), n_surf, n_lane,
           n["buildings"], n["trees"], n["poles"], n["vehicles"]))


main()
