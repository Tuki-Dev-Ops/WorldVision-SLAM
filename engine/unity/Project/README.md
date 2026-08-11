# 바로 열 수 있는 Unity 프로젝트

Unity Hub 에서 **Add > Add project from disk** 로 이 폴더를 고르면 된다.
임포터가 `Assets/Editor/` 에 이미 있으므로, 열린 뒤 메뉴에서
**WorldVision > Import Scene (JSON)** 을 고르고
`results/scene/kitti_00.json` 을 열면 씬이 선다.

`ProjectVersion.txt` 는 6000.0.81f1 을 가리킨다. 다른 버전으로 열면 Hub 가
업그레이드를 물어보는데, 이 프로젝트에는 버전에 묶인 에셋이 없으므로
그대로 진행해도 된다.

`manifest.json` 에는 임포터가 실제로 쓰는 모듈만 넣었다 - JSON 직렬화,
IMGUI(에디터 메뉴), 물리(콜라이더), UI. 나머지는 Unity 가 처음 열 때
자기 기본값으로 채운다.
