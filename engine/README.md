# 인식한 장면을 게임 엔진으로

뷰어가 화면에 그리는 것은 이미 장면 서술이다. 집은 중심·방향·치수·지붕
높이이고, 나무는 자리·높이·수관 반지름이며, 차량은 변환과 크기와 클래스다.
그것을 파일로 내보내면 Unity 나 Unreal 이 받아 재질과 조명을 입힐 수 있다.

## 두 가지 형식

```
wme_bench_viewer --seq 0 --frame 200 \
    --export-scene results/scene/kitti_00.glb \
    --export-json  results/scene/kitti_00.json \
    --screenshot   /tmp/shot.png
```

**`.glb`** — 삼각형을 굳혀 담은 것. 두 엔진이 별도 임포터 없이 읽는다
(Unreal 은 기본 지원, Unity 는 glTFast 또는 UnityGLTF 패키지가 필요하다).
장면을 눈으로 확인하거나 배경으로 한 번 놓을 때 쓴다.

**`.json`** — 무엇이 어디에 얼마만 한가를 적은 것. 50 KB 남짓이다.
아래 임포터가 이것을 읽어 오브젝트를 놓는다.

**둘의 차이는 크기가 아니라 종류다.** GLB 는 집 하나가 커다란 메시의 일부라
충돌체도 재질도 따로 줄 수 없고, 차를 움직이거나 건물을 지울 수도 없다.
JSON 으로 세우면 엔진이 아는 오브젝트가 되므로 그다음은 엔진의 일이 된다.

## Unity

`engine/unity/Editor/WorldVisionSceneImporter.cs` 를 프로젝트의 아무
`Editor` 폴더에 넣고 메뉴에서 **WorldVision > Import Scene (JSON)**.

기본 도형(Cube/Sphere/Cylinder)만 쓰므로 에셋을 따로 받을 필요가 없고,
셰이더는 URP · HDRP · Built-in 순으로 찾는다. 프리팹으로 바꾸려면 생성된
GameObject 를 교체하면 된다.

좌표는 오른손 +y up 으로 나가고 Unity 는 왼손 +y up 이라 z 를 뒤집는다.
방향 벡터도 같은 변환을 받는다 - 위치만 바꾸면 집이 제자리에 서되 엉뚱한
쪽을 본다.

## Unreal

Output Log (Window > Output Log) 의 Cmd 입력줄에서:

```
py "D:/Program Files/Slam-Model/engine/unreal/import_worldvision_scene.py"
```

다른 장면을 쓰려면 환경변수 `WV_SCENE` 에 경로를 넣는다. 액터는 아웃라이너의
`WorldVision/` 아래에 클래스별 폴더로 묶인다.

언리얼은 왼손계에 **+z 가 위** 이고 단위가 센티미터다. 축을 바꾸고 100 을
곱한다 - 축만 바꾸고 단위를 잊으면 장면이 100 배 작게 들어와 원점의 점처럼
보인다.

## 장면 서술 형식

```json
{
  "format": "worldvision-scene/1",
  "sequence": "kitti_00", "frame": 201,
  "up": "+y", "handedness": "right", "unit": "meter",
  "ego": [x, y, z],
  "buildings": [{"center": [..], "forward": [..], "length": 7.4,
                 "width": 1.4, "height": 3.7, "roof": 0.8, "range": 23.9}],
  "trees":     [{"foot": [..], "height": 6.5, "canopy": 1.8}],
  "poles":     [{"foot": [..], "height": 4.0}],
  "vehicles":  [{"class": "car", "center": [..], "forward": [..],
                 "size": [w, h, l], "seen": 7, "moving": false}]
}
```

`range` 는 그 집을 **가장 가까이서 본 거리** 다. 스테레오 깊이 오차는 거리의
제곱으로 커지므로 (30 m 에서 약 1.2 m), 먼 것만 걸러 쓰고 싶으면 이 값으로
거른다. 뷰어는 30 m 를 넘는 것을 이미 빼고 내보낸다.

`moving` 은 클래스가 아니라 **관측** 에서 나온 판정이다. 주차된 차와 지나간
차를 가를 때 쓴다.

## 확인한 것과 확인하지 못한 것

**GLB** 는 규격에 맞는지 검사했다 - 청크 길이, 선언한 버퍼 크기 대 실제 청크,
모든 접근자의 count 대 bufferView 바이트 길이.

**Unity 임포터는 컴파일된다.** `engine/unity/Check/check.ps1` 이 Unity 없이
그것을 확인한다 - `UnityStubs.cs` 가 임포터가 쓰는 API 만 흉내 내고, 시그니처가
어긋나면 컴파일러가 잡는다. 에디터를 열었을 때 가장 흔히 깨지는 것이 컴파일
에러이므로, 그 경로는 설치 없이도 막을 수 있다. Visual Studio 나 Build Tools 가
있으면 돈다.

**좌표 변환** 은 실제 데이터로 확인했다 (kitti_00 201 프레임):

| 검사 | 결과 |
|---|---|
| 건물 · 나무 · 기둥 · 차량 | 237 · 56 · 1 · 46 |
| NaN / Inf 좌표 | 0 |
| 단위벡터가 아닌 forward | 0 (건물 237, 차량 46 전부) |
| 변환 전후 상호 거리 변화 | 0.000000 m |
| Unity 좌표 범위 | x -86.9..25.2, y -0.3..15.1, z 18.5..255.3 m |
| 치수가 말이 안 되는 건물 | 0 |

거리가 그대로인 것이 중요하다 - z 뒤집기는 거울이라 손 방향만 바뀌고 형상은
보존되어야 한다. 값이 달라졌다면 축을 잘못 섞은 것이다.

**실제 에디터에서 돌려 본 것은 아니다.** Unity 도 Unreal 도 여기 설치되어
있지 않다. Unity 설치는 내려받기까지는 되지만 **라이선스 활성화에 계정 로그인이
필요하고**, `-batchmode` 도 라이선스 없이는 멈춘다. 그건 저장소 주인이 할 일이다.

그래서 위 표가 대신하는 것은 "임포트가 성공한다" 가 아니라 "임포트가 실패할
알려진 이유들이 없다" 다. 조명·재질·스케일 감각처럼 눈으로 봐야 아는 것은
에디터에서 확인해야 한다.
