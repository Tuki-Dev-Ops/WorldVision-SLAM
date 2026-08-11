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

    기본 경로는 아래 SCENE_JSON 이다. 다른 파일을 쓰려면 그 값을 바꾸거나
    환경변수 WV_SCENE 에 경로를 넣는다.

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
import os
import unreal

SCENE_JSON = os.environ.get(
    "WV_SCENE",
    r"D:/Program Files/Slam-Model/results/scene/kitti_00.json")

M = 100.0  # 미터 -> 센티미터

# 엔진 기본 도형. 크기가 100 cm 인 정육면체/구/원기둥이라 스케일 1 이 1 m 다.
CUBE = "/Engine/BasicShapes/Cube.Cube"
SPHERE = "/Engine/BasicShapes/Sphere.Sphere"
CYLINDER = "/Engine/BasicShapes/Cylinder.Cylinder"


def to_ue(v):
    """오른손 +y up (m) -> 언리얼 왼손 +z up (cm)."""
    return unreal.Vector(-v[2] * M, v[0] * M, v[1] * M)


def yaw_from(fwd):
    """진행 방향에서 요각을 뽑는다. 지면 위 회전만 쓰므로 yaw 하나면 된다."""
    x = -fwd[2]
    y = fwd[0]
    if abs(x) < 1e-6 and abs(y) < 1e-6:
        return unreal.Rotator(0.0, 0.0, 0.0)
    import math
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


def main():
    if not os.path.exists(SCENE_JSON):
        unreal.log_error("장면 파일이 없다: %s" % SCENE_JSON)
        return
    with open(SCENE_JSON, "r", encoding="utf-8") as f:
        s = json.load(f)
    if s.get("format") != "worldvision-scene/1":
        unreal.log_error("형식이 맞지 않는다: %s" % s.get("format"))
        return

    n = {"buildings": 0, "trees": 0, "poles": 0, "vehicles": 0}

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

    unreal.log("WorldVision: %s 프레임 %s - 건물 %d, 나무 %d, 기둥 %d, 차량 %d"
               % (s.get("sequence"), s.get("frame"), n["buildings"],
                  n["trees"], n["poles"], n["vehicles"]))


main()
