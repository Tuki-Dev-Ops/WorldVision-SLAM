// WorldVision-SLAM 장면을 Unity 씬으로 세운다.
//
// GLB 를 그냥 끌어다 놓아도 보이기는 한다. 다만 그것은 굳은 삼각형 덩이라
// 집 하나가 벽이 아니라 커다란 메시의 일부이고, 충돌체도 재질도 따로 줄 수
// 없으며 차를 움직이거나 건물을 지울 수도 없다.
//
// 이 스크립트는 --export-json 이 낸 **매개변수 목록** 을 읽어 오브젝트를
// 놓는다. 그때부터는 Unity 가 아는 GameObject 라, 프리팹으로 바꾸든 물리를
// 붙이든 그다음은 엔진의 일이 된다.
//
// 사용: 메뉴 WorldVision > Import Scene (JSON)
//
// 좌표에 대하여
// -------------
// 내보낸 좌표는 오른손계에 +y 가 위다 (glTF 규약). Unity 는 왼손계에 +y 가
// 위이므로 z 를 뒤집으면 맞는다. 방향 벡터도 같은 변환을 받아야 한다 -
// 위치만 바꾸면 집이 제자리에 서되 엉뚱한 쪽을 본다.

using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace WorldVision
{
    [Serializable] public class Building
    {
        public float[] center;
        public float[] forward;
        public float length;
        public float width;
        public float height;
        public float roof;
        public float range;
    }

    [Serializable] public class Tree
    {
        public float[] foot;
        public float height;
        public float canopy;
    }

    [Serializable] public class Pole
    {
        public float[] foot;
        public float height;
    }

    [Serializable] public class Vehicle
    {
        public string cls;          // JSON 의 "class" 는 C# 예약어라 이름을 바꿔 받는다
        public float[] center;
        public float[] forward;
        public float[] size;
        public int seen;
        public bool moving;
    }

    [Serializable] public class Road
    {
        public float cell;
        // JsonUtility 는 float[][] 를 못 읽는다. 타일은 [x, y, z, 밝기] 인데
        // 중첩 배열이라 직접 훑는다 - ReadRoad 참조.
    }

    [Serializable] public class SceneFile
    {
        public string format;
        public string sequence;
        public int frame;
        public float[] ego;
        public Building[] buildings;
        public Tree[] trees;
        public Pole[] poles;
        public Vehicle[] vehicles;
    }

    // 지표면 타일 하나. [x, y, z, 밝기, 종류]
    public struct SurfaceTile
    {
        public Vector3 p;
        public float v;     // 0..1 밝기
        public int cls;     // 0 도로 / 1 인도 / 2 잔디 / 3 기타
    }

    // 노면 밝기 지도의 놓임새. 차선은 기하가 아니라 무늬라 이쪽으로 온다.
    public class RoadMap
    {
        public string image;
        public float cell = 0.1f;
        public int width, height;
        public Vector3 origin, axisU, axisV;
        public bool valid;
    }

    public static class SceneImporter
    {
        // 오른손 +y up  ->  Unity 왼손 +y up
        static Vector3 P(float[] v)
        {
            return new Vector3(v[0], v[1], -v[2]);
        }

        static Quaternion Facing(float[] fwd)
        {
            Vector3 f = P(fwd);
            if (f.sqrMagnitude < 1e-6f) return Quaternion.identity;
            return Quaternion.LookRotation(f.normalized, Vector3.up);
        }

        // 생성한 메시와 재질은 **에셋으로 저장한다.**
        //
        // 처음에는 씬에 그대로 직렬화했는데, 타일이 만 이천에서 이만 육천으로
        // 늘자 빌드된 실행 파일이 "level0 is corrupted / Position out of
        // bounds" 로 죽었다. 씬 파일 하나에 메시 수십만 정점을 밀어 넣은
        // 결과다. 에셋으로 빼면 각자 제 파일로 직렬화되고 빌드도 가벼워진다.
        const string kAssetDir = "Assets/WorldVision";

        // **폴더는 AssetDatabase 를 통해 만들어야 한다.**
        //
        // Directory.CreateDirectory 로 만든 폴더는 디스크에는 있지만
        // 데이터베이스가 모르므로 CreateAsset 이 조용히 실패한다. 그러면
        // 메시가 에셋이 안 되고 씬에 그대로 직렬화되어, 빌드는 성공하는데
        // 실행할 때 "level0 is corrupted" 로 죽는다. 두 번 그렇게 잃었다.
        static void EnsureAssetDir()
        {
            if (!AssetDatabase.IsValidFolder(kAssetDir))
            {
                AssetDatabase.CreateFolder("Assets", "WorldVision");
            }
        }

        static T Persist<T>(T obj, string name) where T : UnityEngine.Object
        {
            if (obj == null) return null;
            EnsureAssetDir();
            string path = kAssetDir + "/" + name + ".asset";
            AssetDatabase.DeleteAsset(path);
            AssetDatabase.CreateAsset(obj, path);
            // 조용히 실패하면 안 된다. 여기서 안 잡히면 실행할 때 잡힌다.
            if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(path) == null)
            {
                Debug.LogError("WorldVision: 에셋을 못 만들었다 - " + path);
            }
            return obj;
        }

        static Material MakeMaterial(string name, Color c)
        {
            // 렌더 파이프라인마다 표준 셰이더 이름이 다르다. URP/HDRP 를
            // 쓰는 프로젝트에서 Standard 를 찾으면 분홍색이 되므로 순서대로
            // 찾아본다.
            Shader sh = Shader.Find("Universal Render Pipeline/Lit")
                     ?? Shader.Find("HDRP/Lit")
                     ?? Shader.Find("Standard");
            var m = new Material(sh) { name = name };
            m.color = c;
            // 같은 재질을 쓰는 상자가 수백 개다. 인스턴싱을 켜면 드로콜이
            // 그만큼 묶인다 - 내장 GPU 에서 이 차이가 크다.
            return Persist(m, name);
        }

        static GameObject Box(Transform parent, string name, Vector3 pos,
                              Quaternion rot, Vector3 size, Material mat)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.position = pos;
            go.transform.rotation = rot;
            go.transform.localScale = size;
            go.GetComponent<Renderer>().sharedMaterial = mat;
            return go;
        }

        // 배치 모드 진입점.
        //
        //   Unity.exe -batchmode -quit
        //             -projectPath <프로젝트>
        //             -executeMethod WorldVision.SceneImporter.ImportFromCommandLine
        //             -wvScene <장면.json>
        //
        // 에디터를 띄우지 않고 씬을 세울 때 쓴다. 라이선스는 여전히
        // 필요하다 - -batchmode 도 활성화된 라이선스 없이는 시작하지 않는다.
        public static void ImportFromCommandLine()
        {
            string path = null;
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i + 1 < args.Length; ++i)
            {
                if (args[i] == "-wvScene") { path = args[i + 1]; break; }
            }
            if (string.IsNullOrEmpty(path))
            {
                Debug.LogError("WorldVision: -wvScene <경로> 가 필요하다");
                EditorApplication.Exit(2);
                return;
            }
            if (!Build(path))
            {
                EditorApplication.Exit(1);
                return;
            }
            // 배치 모드에서는 씬을 저장해야 결과가 남는다.
            EditorSceneManager.SaveScene(
                EditorSceneManager.GetActiveScene(),
                "Assets/WorldVisionScene.unity");
            EditorApplication.Exit(0);
        }

        // 세운 씬을 위에서 한 장 찍는다.
        //
        //   Unity.exe -batchmode -quit -projectPath <프로젝트>
        //             -executeMethod WorldVision.SceneImporter.RenderFromCommandLine
        //             -wvScene <장면.json> -wvShot <출력.png>
        //
        // **-nographics 를 붙이면 안 된다.** 그래픽 장치가 없으면 렌더가
        // 빈 그림을 내놓는다 - 임포트만 할 때와 다른 점이다.
        //
        // 항공뷰로 찍는 이유는 이것이 확인하려는 것이 배치이기 때문이다.
        // 조명이나 재질은 눈으로 볼 문제이고, 건물이 길 양옆에 제자리로
        // 섰는지는 위에서 보면 한눈에 갈린다.
        public static void RenderFromCommandLine()
        {
            string scene = null, shot = null;
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i + 1 < args.Length; ++i)
            {
                if (args[i] == "-wvScene") scene = args[i + 1];
                else if (args[i] == "-wvShot") shot = args[i + 1];
            }
            if (string.IsNullOrEmpty(scene) || string.IsNullOrEmpty(shot))
            {
                Debug.LogError("WorldVision: -wvScene 과 -wvShot 이 필요하다");
                EditorApplication.Exit(2);
                return;
            }
            if (!Build(scene)) { EditorApplication.Exit(1); return; }

            var sun = new GameObject("Sun").AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.intensity = 1.1f;
            sun.transform.rotation = Quaternion.Euler(52f, -30f, 0f);

            // 씬 전체가 들어오도록 경계에서 카메라 높이를 정한다. 고정
            // 높이를 쓰면 시퀀스마다 잘리거나 점처럼 작아진다.
            Bounds bb = new Bounds(Vector3.zero, Vector3.zero);
            bool got = false;
            foreach (var r in UnityEngine.Object.FindObjectsByType<Renderer>(
                         FindObjectsSortMode.None))
            {
                if (!got) { bb = r.bounds; got = true; }
                else bb.Encapsulate(r.bounds);
            }
            var camGo = new GameObject("ShotCamera");
            var cam = camGo.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.05f, 0.06f, 0.08f);
            float span = Mathf.Max(bb.size.x, bb.size.z);
            camGo.transform.position = bb.center + Vector3.up * (span * 0.75f + 30f);
            camGo.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
            cam.farClipPlane = span * 3f + 200f;

            const int W = 1600, H = 1000;
            var rt = new RenderTexture(W, H, 24);
            cam.targetTexture = rt;
            cam.Render();
            RenderTexture.active = rt;
            var tex = new Texture2D(W, H, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, W, H), 0, 0);
            tex.Apply();
            RenderTexture.active = null;
            cam.targetTexture = null;
            File.WriteAllBytes(shot, tex.EncodeToPNG());
            Debug.Log("WorldVision: 렌더 저장 " + shot + "  범위 " + bb.size);
            EditorApplication.Exit(0);
        }

        // 걸어 다닐 수 있는 실행 파일을 만든다.
        //
        //   Unity.exe -batchmode -quit -projectPath <프로젝트>
        //             -executeMethod WorldVision.SceneImporter.BuildPlayerFromCommandLine
        //             -wvScene <장면.json> -wvOut <출력폴더>
        //
        // 임포트와 렌더까지는 **보기만 하는 것** 이었다. 여기서 나오는 것은
        // 실행하면 그 안에 서 있게 되는 프로그램이다 - 노면에 붙인
        // MeshCollider 와 건물의 BoxCollider 가 그때 비로소 쓰인다.
        public static void BuildPlayerFromCommandLine()
        {
            string scene = null, outDir = null;
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i + 1 < args.Length; ++i)
            {
                if (args[i] == "-wvScene") scene = args[i + 1];
                else if (args[i] == "-wvOut") outDir = args[i + 1];
            }
            if (string.IsNullOrEmpty(scene) || string.IsNullOrEmpty(outDir))
            {
                Debug.LogError("WorldVision: -wvScene 과 -wvOut 이 필요하다");
                EditorApplication.Exit(2);
                return;
            }
            if (!Build(scene)) { EditorApplication.Exit(1); return; }

            Dress(scene);

            // 창 제목이 "Project" 로 나오면 무엇을 실행한 것인지 알 수 없다.
            PlayerSettings.companyName = "WorldVision";
            PlayerSettings.productName = "WorldVision-SLAM";
            PlayerSettings.defaultScreenWidth = 1600;
            PlayerSettings.defaultScreenHeight = 900;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.resizableWindow = true;
            PlayerSettings.runInBackground = true;

            const string scenePath = "Assets/WorldVisionScene.unity";
            EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(), scenePath);
            EditorBuildSettings.scenes = new[] {
                new EditorBuildSettingsScene(scenePath, true) };

            Directory.CreateDirectory(outDir);
            var opt = new BuildPlayerOptions {
                scenes = new[] { scenePath },
                locationPathName = Path.Combine(outDir, "WorldVision.exe"),
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None,
            };
            var report = BuildPipeline.BuildPlayer(opt);
            var sum = report.summary;
            Debug.Log(string.Format(
                "WorldVision: 빌드 {0}  {1} 바이트  {2}",
                sum.result, sum.totalSize, opt.locationPathName));
            EditorApplication.Exit(sum.result == BuildResult.Succeeded ? 0 : 1);
        }

        // 해와 하늘과 사람을 놓는다. 임포터가 세우는 것은 형상뿐이라,
        // 이것 없이는 검은 화면에 회색 덩어리만 뜬다.
        static void Dress(string scenePath)
        {
            var sun = new GameObject("Sun").AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.intensity = 1.15f;
            sun.color = new Color(1f, 0.97f, 0.9f);
            sun.shadows = LightShadows.Soft;
            sun.transform.rotation = Quaternion.Euler(48f, -35f, 0f);

            RenderSettings.ambientLight = new Color(0.38f, 0.42f, 0.5f);
            RenderSettings.fog = true;
            RenderSettings.fogColor = new Color(0.62f, 0.68f, 0.78f);
            RenderSettings.fogMode = FogMode.Linear;
            RenderSettings.fogStartDistance = 60f;
            RenderSettings.fogEndDistance = 320f;

            // **자차가 섰던 자리에 세운다.** 장면의 한가운데가 아니다 - 거기는
            // 건물 안일 수도 있다. ego 는 실제로 차가 지나간 자리이므로
            // 노면 위인 것이 보장된다.
            Vector3 start = new Vector3(0f, 3f, 0f);
            var ego = GameObject.Find("Ego");
            // 3 m 위에서 떨어뜨린다. 자차 높이에 딱 맞춰 놓으면 그 자리에
            // 주차된 차 안에서 시작하는 일이 생긴다.
            if (ego != null) start = ego.transform.position + Vector3.up * 3f;

            // **길을 따라 보게 세운다.** 처음 보이는 것이 담벼락이면 무엇을
            // 세운 것인지 알 수 없다. 주변 노면 타일의 주축이 곧 길의 방향
            // 이므로 그것을 구해 쓴다 - 고정 방향은 시퀀스마다 어긋난다.
            float yaw = 0f;
            {
                var road = GameObject.Find("Road");
                if (road != null)
                {
                    var mf = road.GetComponent<MeshFilter>();
                    if (mf != null && mf.sharedMesh != null)
                    {
                        // 2 x 2 공분산의 주고유벡터. 평면 문제라 3 차원 분해가
                        // 필요 없다.
                        double sxx = 0, sxz = 0, szz = 0; int n = 0;
                        var vs = mf.sharedMesh.vertices;
                        for (int i = 0; i < vs.Length; i += 4)
                        {
                            Vector3 d = vs[i] - start;
                            d.y = 0f;
                            if (d.sqrMagnitude > 900f) continue;   // 30 m
                            sxx += d.x * d.x; sxz += d.x * d.z; szz += d.z * d.z;
                            ++n;
                        }
                        if (n >= 8)
                        {
                            double tr = sxx + szz;
                            double det = sxx * szz - sxz * sxz;
                            double l = 0.5 * tr + Math.Sqrt(
                                Math.Max(0.0, 0.25 * tr * tr - det));
                            Vector3 axis = Math.Abs(sxz) > 1e-9
                                ? new Vector3((float)(l - szz), 0f, (float)sxz).normalized
                                : (sxx >= szz ? Vector3.right : Vector3.forward);
                            // 주축은 부호가 없다. 노면이 더 많이 뻗은 쪽으로 돌린다.
                            int fwd = 0, back = 0;
                            for (int i = 0; i < vs.Length; i += 4)
                            {
                                Vector3 d = vs[i] - start; d.y = 0f;
                                if (d.sqrMagnitude > 900f) continue;
                                if (Vector3.Dot(d, axis) > 0f) ++fwd; else ++back;
                            }
                            if (back > fwd) axis = -axis;
                            yaw = Quaternion.LookRotation(axis, Vector3.up).eulerAngles.y;
                        }
                    }
                }
            }

            var body = new GameObject("Player");
            body.transform.position = start;
            body.transform.rotation = Quaternion.Euler(0f, yaw, 0f);
            var cc = body.AddComponent<CharacterController>();
            cc.height = 1.75f;
            cc.radius = 0.35f;
            cc.center = new Vector3(0f, 0.875f, 0f);
            cc.slopeLimit = 55f;
            cc.stepOffset = 0.45f;

            var eye = new GameObject("Eye");
            eye.transform.SetParent(body.transform, false);
            eye.transform.localPosition = new Vector3(0f, 1.62f, 0f);
            var cam = eye.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.Skybox;
            cam.backgroundColor = new Color(0.58f, 0.66f, 0.78f);
            cam.nearClipPlane = 0.12f;
            cam.farClipPlane = 400f;
            cam.fieldOfView = 68f;
            eye.tag = "MainCamera";

            body.AddComponent<Player>();

            // **자동 주행.** 달릴 자리는 지어내지 않는다 - 궤적이 곧 정답
            // 주행이고, 그 위를 도는 것만으로도 노면이 이어져 있는지 차선이
            // 코너에서 끊기는지가 드러난다.
            var carGo = new GameObject("EgoCar");
            var bodyBox = GameObject.CreatePrimitive(PrimitiveType.Cube);
            bodyBox.name = "Body";
            bodyBox.transform.SetParent(carGo.transform, false);
            bodyBox.transform.localPosition = new Vector3(0f, 0.72f, 0f);
            bodyBox.transform.localScale = new Vector3(1.78f, 0.72f, 4.30f);
            var matEgo = MakeMaterial("wv_ego", new Color(0.86f, 0.32f, 0.20f));
            bodyBox.GetComponent<Renderer>().sharedMaterial = matEgo;
            UnityEngine.Object.DestroyImmediate(bodyBox.GetComponent<BoxCollider>());
            var cab = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cab.name = "Cabin";
            cab.transform.SetParent(carGo.transform, false);
            cab.transform.localPosition = new Vector3(0f, 1.30f, -0.25f);
            cab.transform.localScale = new Vector3(1.58f, 0.62f, 2.10f);
            cab.GetComponent<Renderer>().sharedMaterial =
                MakeMaterial("wv_ego_glass", new Color(0.16f, 0.22f, 0.28f));
            UnityEngine.Object.DestroyImmediate(cab.GetComponent<BoxCollider>());

            var dir = new GameObject("Director").AddComponent<Director>();
            dir.route = UnityEngine.Object.FindFirstObjectByType<Route>();
            dir.stats = UnityEngine.Object.FindFirstObjectByType<Stats>();
            dir.cam = cam;
            dir.car = carGo.transform;
            dir.free = body.GetComponent<Player>();
        }

        // 런타임에 짓는 실행 파일을 만든다.
        //
        //   Unity.exe -batchmode -quit -projectPath <프로젝트>
        //             -executeMethod WorldVision.SceneImporter.BuildSimFromCommandLine
        //             -wvScene <장면.json> -wvOut <출력폴더>
        //
        // 씬에는 카메라와 빛과 부트 오브젝트만 넣는다. 지도는 실행할 때
        // StreamingAssets 에서 읽어 짓는다 - 씬에 다 넣었더니 큰 장면에서
        // 빌드가 실행 중에 죽었고, 그 원인을 좁히는 데 오래 걸렸다.
        public static void BuildSimFromCommandLine()
        {
            string scene = null, outDir = null;
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i + 1 < args.Length; ++i)
            {
                if (args[i] == "-wvScene") scene = args[i + 1];
                else if (args[i] == "-wvOut") outDir = args[i + 1];
            }
            if (string.IsNullOrEmpty(scene) || string.IsNullOrEmpty(outDir))
            {
                Debug.LogError("WorldVision: -wvScene 과 -wvOut 이 필요하다");
                EditorApplication.Exit(2);
                return;
            }

            // 데이터를 StreamingAssets 로. 여기 있는 것은 직렬화되지 않고
            // 파일 그대로 복사되므로 씬 크기와 무관하다.
            // **데이터는 여기서 옮기지 않는다.**
            //
            // 예전에는 장면 파일을 Assets/StreamingAssets 로 복사해 빌드에
            // 포장했다. 두 가지가 나빴다. 시퀀스를 여럿 담으니 빌드가 커져
            // 실행되지 않았고, 파일 API 로 그 폴더를 갈아엎은 직후의
            // AssetDatabase.Refresh 가 방금 저장한 씬을 흔들어 - 씬 없는
            // 빌드가 나왔다. 데이터는 build.ps1 이 실행 파일 옆 wvdata 에
            // 넣고, 런타임이 거기서 읽는다.
            string baseName = Path.GetFileNameWithoutExtension(scene);

            var root = new GameObject("WorldVision");
            var boot = root.AddComponent<Boot>();
            boot.sceneName = baseName;

            var sun = new GameObject("Sun").AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.intensity = 0.9f;
            sun.color = new Color(1f, 0.97f, 0.92f);
            sun.shadows = LightShadows.None;
            sun.transform.rotation = Quaternion.Euler(50f, -35f, 0f);
            RenderSettings.ambientLight = new Color(0.42f, 0.46f, 0.55f);
            RenderSettings.fog = false;

            var body = new GameObject("Player");
            body.transform.position = new Vector3(0f, 3f, 0f);
            var cc = body.AddComponent<CharacterController>();
            cc.height = 1.75f; cc.radius = 0.35f;
            cc.center = new Vector3(0f, 0.875f, 0f);
            cc.slopeLimit = 55f; cc.stepOffset = 0.45f;

            var eye = new GameObject("Eye");
            eye.transform.SetParent(body.transform, false);
            eye.transform.localPosition = new Vector3(0f, 1.62f, 0f);
            var cam = eye.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.02f, 0.025f, 0.035f);
            cam.nearClipPlane = 0.12f;
            cam.farClipPlane = 600f;
            cam.fieldOfView = 60f;
            eye.tag = "MainCamera";
            body.AddComponent<Player>();

            var carGo = new GameObject("EgoCar");
            var bodyBox = GameObject.CreatePrimitive(PrimitiveType.Cube);
            bodyBox.name = "Body";
            bodyBox.transform.SetParent(carGo.transform, false);
            bodyBox.transform.localPosition = new Vector3(0f, 0.72f, 0f);
            bodyBox.transform.localScale = new Vector3(1.78f, 0.72f, 4.30f);
            bodyBox.GetComponent<Renderer>().sharedMaterial =
                MakeMaterial("wv_ego", new Color(0.90f, 0.90f, 0.92f));
            UnityEngine.Object.DestroyImmediate(bodyBox.GetComponent<BoxCollider>());
            var cabin = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cabin.name = "Cabin";
            cabin.transform.SetParent(carGo.transform, false);
            cabin.transform.localPosition = new Vector3(0f, 1.30f, -0.25f);
            cabin.transform.localScale = new Vector3(1.58f, 0.62f, 2.10f);
            cabin.GetComponent<Renderer>().sharedMaterial =
                MakeMaterial("wv_ego_glass", new Color(0.20f, 0.26f, 0.34f));
            UnityEngine.Object.DestroyImmediate(cabin.GetComponent<BoxCollider>());

            var dir = root.AddComponent<Director>();
            dir.boot = boot;
            dir.cam = cam;
            dir.car = carGo.transform;
            dir.free = body.GetComponent<Player>();

            // 선 그리기는 카메라에 붙어야 OnRenderObject 가 그 카메라의
            // 렌더 중에 불린다.
            var lines = eye.AddComponent<Lines>();
            lines.boot = boot;
            lines.car = carGo.transform;

            PlayerSettings.companyName = "WorldVision";
            PlayerSettings.productName = "WorldVision-SLAM";
            PlayerSettings.defaultScreenWidth = 1600;
            PlayerSettings.defaultScreenHeight = 900;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.resizableWindow = true;
            PlayerSettings.runInBackground = true;
            // **셰이더는 씬에서 참조되게 한다.**
            //
            // 예전에는 그래픽 설정의 "항상 포함" 목록에 넣었는데, 그것은
            // 빌드 직전에 ProjectSettings 를 고치는 일이다. 씬 저장과
            // 에셋 저장이 그 사이에서 엉키면서 실행되지 않는 빌드가 나왔다.
            // 눈에 안 띄는 작은 메시 하나가 그 재질을 들고 있으면, 셰이더는
            // 평범한 의존성으로 따라 들어간다.
            {
                var keep = new GameObject("ShaderKeepAlive");
                keep.transform.SetParent(root.transform, false);
                keep.transform.localScale = new Vector3(0.001f, 0.001f, 0.001f);
                var mf = keep.AddComponent<MeshFilter>();
                var m = new Mesh();
                m.name = "keepalive";
                m.SetVertices(new List<Vector3> { Vector3.zero });
                m.SetIndices(new[] { 0 }, MeshTopology.Points, 0);
                mf.sharedMesh = m;
                var sh = Shader.Find("WorldVision/Point");
                if (sh == null) Debug.LogWarning("WorldVision: 점 셰이더 없음");
                keep.AddComponent<MeshRenderer>().sharedMaterial =
                    Persist(new Material(sh != null ? sh : Shader.Find("Unlit/Color"))
                            { name = "wv_point_keep" }, "wv_point_keep");
                Persist(m, "SM_keepalive");
            }

            const string scenePath = "Assets/WorldVisionSim.unity";
            EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(), scenePath);
            // **저장한 씬이 에셋 데이터베이스에 들어왔는지 확인하고 짓는다.**
            //
            // Library 를 지우고 나면 저장 직후의 씬을 데이터베이스가 아직
            // 모른다. 그 상태로 BuildPlayer 를 부르면 **조용히 씬 없이**
            // 빌드된다 - 결과물에 level0 이 아예 없고, 실행하면 "level0 is
            // corrupted" 로 죽는다. 파일이 깨진 것이 아니라 없는 것이었다.
            AssetDatabase.Refresh();
            if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(scenePath) == null)
            {
                Debug.LogError("WorldVision: 씬이 에셋으로 안 잡혔다 - " + scenePath);
                EditorApplication.Exit(3);
                return;
            }
            // **붙은 스크립트가 전부 에셋으로 풀리는지 본다.**
            //
            // 이것이 "level0 is corrupted" 의 원인이다 - 아래 VerifySceneScripts
            // 주석 참조. 여기서 막지 않으면 빌드는 Succeeded 를 내고 실행할 때
            // 죽는다. 조용히 넘기지 않는다.
            if (!VerifySceneScripts(scenePath))
            {
                EditorApplication.Exit(4);
                return;
            }
            EditorBuildSettings.scenes = new[] {
                new EditorBuildSettingsScene(scenePath, true) };

            Directory.CreateDirectory(outDir);
            var opt = new BuildPlayerOptions {
                scenes = new[] { scenePath },
                locationPathName = Path.Combine(outDir, "WorldVision.exe"),
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None,
            };
            var report = BuildPipeline.BuildPlayer(opt);
            var sum = report.summary;
            Debug.Log(string.Format("WorldVision: 빌드 {0}  {1} 바이트  {2}",
                sum.result, sum.totalSize, opt.locationPathName));
            EditorApplication.Exit(sum.result == BuildResult.Succeeded ? 0 : 1);
        }

        // **저장한 씬의 MonoBehaviour 가 전부 스크립트 에셋을 가리키는지 본다.**
        //
        // 성한 참조는 이렇게 쓰인다:
        //
        //     m_Script: {fileID: 11500000, guid: c5dd9132..., type: 3}
        //
        // guid 가 없는 것은 씬 안의 없는 오브젝트를 가리키는 끊긴 참조다:
        //
        //     m_Script: {fileID: 574151018}
        //
        // 이렇게 나오는 이유는 **한 .cs 파일이 MonoBehaviour 를 둘 이상 담고
        // 있을 때** 다. 파일 하나에 MonoScript 는 하나뿐이라 (fileID 11500000)
        // 나머지 클래스는 가리킬 에셋이 없다. AddComponent 는 에디터 메모리
        // 안에서 멀쩡히 되므로 여기까지는 아무 일도 없는 것처럼 보이고, 씬을
        // **직렬화할 때** 비로소 끊긴다.
        //
        // 그렇게 나온 level0 은 스크립트를 풀 수 없는 컴포넌트를 담은 채로
        // 빌드되고, 실행하면 "level0 is corrupted / Position out of bounds" 가
        // 된다. Unity 는 빌드 중에 "Script attached to '...' is missing or no
        // valid script is attached" 를 로그에 남기지만 그것은 경고일 뿐이라
        // 빌드가 멈추지 않는다 - 그래서 하루 종일 간헐적으로 보였다.
        //
        // 텍스트로 확인하는 이유는, 실제로 배에 실리는 것이 이 파일이기
        // 때문이다. 메모리 상의 컴포넌트는 멀쩡하다.
        static bool VerifySceneScripts(string scenePath)
        {
            string full = Path.Combine(
                Path.GetDirectoryName(Application.dataPath) ?? ".", scenePath);
            if (!File.Exists(full))
            {
                Debug.LogError("WorldVision: 씬 파일이 없다 - " + full);
                return false;
            }
            string[] lines = File.ReadAllLines(full);

            // fileID -> GameObject 이름. 끊긴 컴포넌트가 어느 오브젝트에
            // 붙어 있는지 말해 주지 않으면 찾는 데 또 하루가 든다.
            var names = new Dictionary<string, string>();
            for (int i = 0; i < lines.Length; ++i)
            {
                if (lines[i] != "GameObject:") continue;
                string id = FileIdOfBlock(lines, i);
                if (id == null) continue;
                for (int j = i + 1; j < lines.Length && !lines[j].StartsWith("---"); ++j)
                {
                    if (!lines[j].StartsWith("  m_Name: ")) continue;
                    names[id] = lines[j].Substring(10).Trim();
                    break;
                }
            }

            int bad = 0;
            for (int i = 0; i < lines.Length; ++i)
            {
                string t = lines[i].TrimStart();
                if (!t.StartsWith("m_Script:")) continue;
                if (t.Contains("guid:")) continue;

                // 어느 오브젝트인가. 같은 블록 안의 m_GameObject 를 거슬러 찾는다.
                string owner = "?";
                for (int j = i; j >= 0 && !lines[j].StartsWith("---"); --j)
                {
                    string u = lines[j].TrimStart();
                    if (!u.StartsWith("m_GameObject:")) continue;
                    string id = Between(u, "fileID: ", "}");
                    if (id != null && names.ContainsKey(id)) owner = names[id];
                    break;
                }
                ++bad;
                Debug.LogError(string.Format(
                    "WorldVision: 스크립트 참조가 끊겼다 - 오브젝트 '{0}', {1} "
                    + "({2}:{3}). 이 컴포넌트의 클래스가 제 파일을 갖고 있지 "
                    + "않다 - .cs 하나에 MonoBehaviour 는 하나만 둔다.",
                    owner, t, scenePath, i + 1));
            }
            if (bad > 0)
            {
                Debug.LogError(string.Format(
                    "WorldVision: 씬에 끊긴 스크립트 {0} 개. 빌드하지 않는다 - "
                    + "이대로 지으면 실행할 때 \"level0 is corrupted\" 로 죽는다.",
                    bad));
                return false;
            }
            Debug.Log("WorldVision: 씬의 스크립트 참조 이상 없음");
            return true;
        }

        // "--- !u!1 &213802038" 에서 213802038 을 꺼낸다. 블록 머리는 바로
        // 윗줄이다.
        static string FileIdOfBlock(string[] lines, int at)
        {
            if (at <= 0) return null;
            int amp = lines[at - 1].IndexOf('&');
            if (amp < 0) return null;
            string s = lines[at - 1].Substring(amp + 1).Trim();
            int sp = s.IndexOf(' ');
            return sp > 0 ? s.Substring(0, sp) : s;
        }

        static string Between(string s, string a, string b)
        {
            int i = s.IndexOf(a);
            if (i < 0) return null;
            i += a.Length;
            int j = s.IndexOf(b, i);
            return j < 0 ? null : s.Substring(i, j - i).Trim();
        }

        // 점군 셰이더는 씬에서 참조되지 않으므로 빌드에서 빠진다. 그래픽
        // 설정의 "항상 포함" 목록에 넣어야 실행할 때 Shader.Find 가 찾는다.
        static void IncludeShader(string name)
        {
            var sh = Shader.Find(name);
            if (sh == null) { Debug.LogWarning("WorldVision: 셰이더 없음 " + name); return; }
            var gs = AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(
                "ProjectSettings/GraphicsSettings.asset");
            if (gs == null) return;
            var so = new SerializedObject(gs);
            var arr = so.FindProperty("m_AlwaysIncludedShaders");
            if (arr == null) return;
            for (int i = 0; i < arr.arraySize; ++i)
                if (arr.GetArrayElementAtIndex(i).objectReferenceValue == sh) return;
            arr.InsertArrayElementAtIndex(arr.arraySize);
            arr.GetArrayElementAtIndex(arr.arraySize - 1).objectReferenceValue = sh;
            so.ApplyModifiedProperties();
            AssetDatabase.SaveAssets();
        }

        [MenuItem("WorldVision/Import Scene (JSON)")]
        public static void Import()
        {
            string path = EditorUtility.OpenFilePanel(
                "WorldVision scene", Application.dataPath, "json");
            if (string.IsNullOrEmpty(path)) return;
            Build(path);
        }

        // **숫자 배열만 직접 훑는다.**
        //
        // JsonUtility 는 중첩 배열(float[][])을 못 읽는다. 나머지는 전부
        // 객체 배열이라 문제가 없고, 지표면만 [x,y,z,밝기,종류] 형태라
        // 여기서만 손으로 읽는다 - 이것 하나 때문에 JSON 라이브러리를
        // 들이지 않는다.
        static float Num(string text, string key, int from, float dflt)
        {
            int at = text.IndexOf(key, from);
            if (at < 0) return dflt;
            int c0 = text.IndexOf(':', at) + 1;
            int c1 = text.IndexOfAny(new[] { ',', '}' }, c0);
            if (c0 <= 0 || c1 < 0) return dflt;
            float v;
            return float.TryParse(text.Substring(c0, c1 - c0).Trim(),
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out v)
                ? v : dflt;
        }

        static Vector3 Vec(string text, string key, int from)
        {
            int at = text.IndexOf(key, from);
            if (at < 0) return Vector3.zero;
            int o = text.IndexOf('[', at), c = text.IndexOf(']', o);
            if (o < 0 || c < 0) return Vector3.zero;
            string[] pp = text.Substring(o + 1, c - o - 1).Split(',');
            if (pp.Length < 3) return Vector3.zero;
            var n = new float[3];
            for (int k = 0; k < 3; ++k)
                float.TryParse(pp[k].Trim(),
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out n[k]);
            // 위치든 방향이든 z 뒤집기는 같다.
            return new Vector3(n[0], n[1], -n[2]);
        }

        static List<SurfaceTile> ReadSurfaces(string text, out float cell)
        {
            var outp = new List<SurfaceTile>();
            cell = 0.5f;
            int at = text.IndexOf("\"surfaces\"");
            if (at < 0) return outp;
            cell = Num(text, "\"cell\"", at, 0.5f);
            int tAt = text.IndexOf('[', text.IndexOf("\"tiles\"", at));
            if (tAt < 0) return outp;
            int i = tAt + 1;
            var num = new float[5];
            while (i < text.Length)
            {
                int open = text.IndexOf('[', i);
                if (open < 0) break;
                int close = text.IndexOf(']', open);
                if (close < 0) break;
                string[] parts = text.Substring(open + 1, close - open - 1).Split(',');
                if (parts.Length >= 5)
                {
                    bool ok = true;
                    for (int k = 0; k < 5; ++k)
                    {
                        ok &= float.TryParse(parts[k].Trim(),
                                System.Globalization.NumberStyles.Float,
                                System.Globalization.CultureInfo.InvariantCulture,
                                out num[k]);
                    }
                    if (ok)
                    {
                        outp.Add(new SurfaceTile {
                            p = new Vector3(num[0], num[1], -num[2]),
                            v = Mathf.Clamp01(num[3] / 255f),
                            cls = Mathf.Clamp((int)num[4], 0, 3) });
                    }
                }
                i = close + 1;
                int nxt = text.IndexOfAny(new[] { '[', ']' }, i);
                if (nxt < 0 || text[nxt] == ']') break;
            }
            return outp;
        }

        static RoadMap ReadRoadMap(string text)
        {
            var m = new RoadMap();
            int at = text.IndexOf("\"road_map\"");
            if (at < 0 || text.IndexOf("null", at, Mathf.Min(24, text.Length - at))
                          == at + 12) return m;
            int q0 = text.IndexOf('"', text.IndexOf("\"image\"", at) + 8);
            int q1 = text.IndexOf('"', q0 + 1);
            if (q0 < 0 || q1 < 0) return m;
            m.image = text.Substring(q0 + 1, q1 - q0 - 1);
            m.cell = Num(text, "\"cell\"", at, 0.1f);
            m.width = (int)Num(text, "\"width\"", at, 0);
            m.height = (int)Num(text, "\"height\"", at, 0);
            m.origin = Vec(text, "\"origin\"", at);
            m.axisU = Vec(text, "\"axis_u\"", at);
            m.axisV = Vec(text, "\"axis_v\"", at);
            m.valid = m.width > 0 && m.height > 0;
            return m;
        }

        // 자차 궤적. 엔진에서 자동으로 달릴 자리다.
        static List<Vector3> ReadTrajectory(string text)
        {
            var outp = new List<Vector3>();
            int at = text.IndexOf("\"trajectory\"");
            if (at < 0) return outp;
            int open = text.IndexOf('[', at);
            int close = text.IndexOf("]]", open);
            if (open < 0 || close < 0) return outp;
            int i = open + 1;
            var n = new float[3];
            while (i < close + 1)
            {
                int a = text.IndexOf('[', i);
                if (a < 0 || a > close) break;
                int b = text.IndexOf(']', a);
                if (b < 0) break;
                string[] pp = text.Substring(a + 1, b - a - 1).Split(',');
                if (pp.Length >= 3)
                {
                    bool ok = true;
                    for (int c = 0; c < 3; ++c)
                        ok &= float.TryParse(pp[c].Trim(),
                                System.Globalization.NumberStyles.Float,
                                System.Globalization.CultureInfo.InvariantCulture,
                                out n[c]);
                    if (ok) outp.Add(new Vector3(n[0], n[1], -n[2]));
                }
                i = b + 1;
            }
            return outp;
        }

        // 차선. **예측한 것이므로 관측한 노면 텍스처와 섞지 않는다.**
        //
        // 구운 밝기 지도는 측정이고 이쪽은 도로 구조에서 세운 모형이다. 따로
        // 두어야 나중에 서로를 채점할 수 있고, 보는 사람도 무엇이 근거이고
        // 무엇이 추정인지 구분할 수 있다.
        static List<List<Vector3>> ReadLanes(string text, out List<string> kinds,
                                             out List<float> widths)
        {
            var outp = new List<List<Vector3>>();
            kinds = new List<string>();
            widths = new List<float>();
            int at = text.IndexOf("\"lanes\"");
            if (at < 0) return outp;
            int i = at;
            while (true)
            {
                int k0 = text.IndexOf("\"kind\"", i);
                if (k0 < 0) break;
                int q0 = text.IndexOf('"', text.IndexOf(':', k0)) + 1;
                int q1 = text.IndexOf('"', q0);
                if (q0 <= 0 || q1 < 0) break;
                string kind = text.Substring(q0, q1 - q0);
                float w = Num(text, "\"width\"", q1, 0.12f);
                int p0 = text.IndexOf("\"points\"", q1);
                if (p0 < 0) break;
                int open = text.IndexOf('[', p0);
                int close = text.IndexOf("]]", open);
                if (open < 0 || close < 0) break;
                var pts = new List<Vector3>();
                int j = open + 1;
                var n = new float[3];
                while (j < close + 1)
                {
                    int a = text.IndexOf('[', j);
                    if (a < 0 || a > close) break;
                    int b = text.IndexOf(']', a);
                    if (b < 0) break;
                    string[] pp = text.Substring(a + 1, b - a - 1).Split(',');
                    if (pp.Length >= 3)
                    {
                        bool ok = true;
                        for (int c = 0; c < 3; ++c)
                            ok &= float.TryParse(pp[c].Trim(),
                                    System.Globalization.NumberStyles.Float,
                                    System.Globalization.CultureInfo.InvariantCulture,
                                    out n[c]);
                        if (ok) pts.Add(new Vector3(n[0], n[1], -n[2]));
                    }
                    j = b + 1;
                }
                if (pts.Count >= 2) { outp.Add(pts); kinds.Add(kind); widths.Add(w); }
                i = close + 2;
                // 배열이 닫혔으면 그만.
                int nx = text.IndexOf("\"kind\"", i);
                int endBracket = text.IndexOf(']', i);
                if (nx < 0 || (endBracket >= 0 && endBracket < nx)) break;
            }
            return outp;
        }

        static void BuildLanes(Transform parent, List<List<Vector3>> lanes,
                               List<string> kinds, List<float> widths)
        {
            if (lanes.Count == 0) return;
            // 노면보다 2 cm 띄운다. 같은 높이면 z-fighting 으로 선이 깜빡인다.
            const float lift = 0.02f;
            var mats = new Material[2];
            mats[0] = MakeMaterial("wv_lane_edge",   new Color(0.90f, 0.90f, 0.88f));
            mats[1] = MakeMaterial("wv_lane_center", new Color(0.94f, 0.88f, 0.45f));
            for (int k = 0; k < 2; ++k)
            {
                if (mats[k].HasProperty("_Glossiness")) mats[k].SetFloat("_Glossiness", 0.05f);
            }

            for (int L = 0; L < lanes.Count; ++L)
            {
                var pts = lanes[L];
                bool center = kinds[L] == "center";
                float hw = Mathf.Max(0.05f, widths[L]) * 0.5f;
                var verts = new List<Vector3>();
                var tris = new List<int>();
                for (int i = 0; i + 1 < pts.Count; ++i)
                {
                    Vector3 a = pts[i] + Vector3.up * lift;
                    Vector3 b = pts[i + 1] + Vector3.up * lift;
                    Vector3 d = b - a; d.y = 0f;
                    if (d.sqrMagnitude < 1e-6f) continue;
                    // 중앙선은 파선으로 놓는다. 실선으로 그으면 추월 금지라는
                    // 뜻이 되는데, 그것은 관측한 것이 아니라 지어낸 규칙이다.
                    if (center && (i % 2) == 1) continue;
                    Vector3 n = new Vector3(-d.z, 0f, d.x).normalized * hw;
                    int v0 = verts.Count;
                    verts.Add(a - n); verts.Add(a + n);
                    verts.Add(b + n); verts.Add(b - n);
                    tris.Add(v0); tris.Add(v0 + 1); tris.Add(v0 + 2);
                    tris.Add(v0); tris.Add(v0 + 2); tris.Add(v0 + 3);
                }
                if (verts.Count == 0) continue;
                var go = new GameObject((center ? "LaneCenter " : "LaneEdge ") + L);
                go.transform.SetParent(parent, false);
                var mesh = new Mesh();
                mesh.SetVertices(verts);
                mesh.SetTriangles(tris, 0);
                mesh.RecalculateNormals();
                mesh.name = "SM_Lane" + L;
                Persist(mesh, mesh.name);
                go.AddComponent<MeshFilter>().sharedMesh = mesh;
                go.AddComponent<MeshRenderer>().sharedMaterial = mats[center ? 1 : 0];
                go.isStatic = true;
            }
        }

        // 표면 종류마다 하나씩. 물성이 다르므로 메시도 갈라야 한다 - 차가
        // 굴러가는 아스팔트와 밟고 서는 보도와 잔디는 같은 재질일 수 없다.
        //
        // 종류 안에서는 타일마다 GameObject 를 두지 않는다. 수천 개가 되어
        // 씬만 무거워지고, 어차피 이어진 한 장의 바닥이다.
        static readonly string[] kSurfName = { "Road", "Sidewalk", "Grass", "Ground" };
        static int _surfTris;
        // 충돌체를 붙일지. 큰 장면에서 빌드가 깨지는 원인을 좁히려고 둔 스위치.
        static bool kSurfCollider =
            System.Environment.GetEnvironmentVariable("WV_NO_COLLIDER") != "1";

        static void BuildSurfaces(Transform parent, List<SurfaceTile> tiles,
                                  float cell, RoadMap rm, Texture2D roadTex,
                                  string projectDir)
        {
            if (tiles.Count == 0) return;

            // **밝기를 정점 색으로 싣지 않는다.** 처음에 그렇게 했더니 화면에
            // 흰 리본 하나만 나왔다 - Lit 셰이더는 COLOR_0 을 읽지 않는다.
            // 포장면은 구워 온 밝기 텍스처를 입히고, 잔디처럼 밝기가 뜻이
            // 없는 표면은 단색으로 둔다.
            var mats = new Material[4];
            mats[0] = MakeMaterial("wv_asphalt",  new Color(0.62f, 0.62f, 0.64f));
            mats[1] = MakeMaterial("wv_sidewalk", new Color(0.74f, 0.72f, 0.69f));
            mats[2] = MakeMaterial("wv_grass",    new Color(0.30f, 0.46f, 0.22f));
            mats[3] = MakeMaterial("wv_ground",   new Color(0.55f, 0.54f, 0.52f));
            if (roadTex != null)
            {
                // 포장면에만 입힌다. 잔디의 밝기는 풀색이 아니라 노출이다.
                mats[0].mainTexture = roadTex;
                mats[1].mainTexture = roadTex;
                // 색은 흰색으로 둔다. 들어 올리는 일은 텍스처를 구울 때
                // 끝냈다 - Standard 의 _Color 는 1 을 넘겨도 잘린다.
                mats[0].color = Color.white;
                mats[1].color = new Color(1.0f, 0.99f, 0.96f);
                if (mats[0].HasProperty("_Glossiness")) mats[0].SetFloat("_Glossiness", 0.22f);
                if (mats[1].HasProperty("_Glossiness")) mats[1].SetFloat("_Glossiness", 0.08f);
            }
            if (mats[2].HasProperty("_Glossiness")) mats[2].SetFloat("_Glossiness", 0.02f);

            float h = cell * 0.5f;
            float spanU = Mathf.Max(1e-3f, rm.width * rm.cell);
            float spanV = Mathf.Max(1e-3f, rm.height * rm.cell);

            // **65,000 정점을 넘기지 않게 쪼갠다.**
            //
            // 도로 하나가 17,957 타일이라 71,828 정점이 되고, 그러면 인덱스가
            // UInt32 로 넘어간다. 그렇게 만든 빌드가 실행할 때 "level0 is
            // corrupted / Position out of bounds" 로 죽었다 - 작은 장면
            // (36,836 정점, UInt16) 은 멀쩡히 떴으므로 경계가 거기다.
            //
            // 쪼개는 편이 어차피 낫다. 메시 하나가 길 전체를 덮으면 절두체
            // 컬링이 통째로 통과시켜, 뒤돌아 서 있어도 도로 전부를 그린다.
            _surfTris = 0;
            const int kMaxTiles = 15000;   // x4 = 60,000 정점
            for (int c = 0; c < 4; ++c)
            {
                var mine = tiles.FindAll(t => t.cls == c);
                if (mine.Count == 0) continue;
                var mat = mats[c];
                int chunks = (mine.Count + kMaxTiles - 1) / kMaxTiles;
                for (int ch = 0; ch < chunks; ++ch)
                {
                    int from = ch * kMaxTiles;
                    int to = Mathf.Min(mine.Count, from + kMaxTiles);
                    var verts = new List<Vector3>((to - from) * 4);
                    var uvs = new List<Vector2>((to - from) * 4);
                    var cols = new List<Color>((to - from) * 4);
                    var tris = new List<int>((to - from) * 6);
                    for (int i = from; i < to; ++i)
                    {
                        var t = mine[i];
                        int b = verts.Count;
                        verts.Add(t.p + new Vector3(-h, 0, -h));
                        verts.Add(t.p + new Vector3( h, 0, -h));
                        verts.Add(t.p + new Vector3( h, 0,  h));
                        verts.Add(t.p + new Vector3(-h, 0,  h));
                        // 밝기를 정점 색으로 싣는다. 포장면은 셰이더가 이것을
                        // 읽고, 잔디는 밝기가 뜻이 없으므로 무시한다.
                        float g = 0.10f + 0.90f * t.v * t.v;
                        var col = new Color(g, g, g * 1.02f);
                        for (int k = 0; k < 4; ++k)
                        {
                            cols.Add(col);
                            Vector3 d = verts[b + k] - rm.origin;
                            uvs.Add(rm.valid
                                ? new Vector2(Vector3.Dot(d, rm.axisU) / spanU,
                                              Vector3.Dot(d, rm.axisV) / spanV)
                                : new Vector2(0.5f, 0.5f));
                        }
                        tris.Add(b); tris.Add(b + 3); tris.Add(b + 2);
                        tris.Add(b); tris.Add(b + 2); tris.Add(b + 1);
                    }

                    var go = new GameObject(chunks > 1
                        ? kSurfName[c] + " " + ch : kSurfName[c]);
                    go.transform.SetParent(parent, false);
                    var mesh = new Mesh();
                    mesh.name = "SM_" + kSurfName[c] + (chunks > 1 ? ch.ToString() : "");
                    mesh.SetVertices(verts);
                    mesh.SetUVs(0, uvs);
                    mesh.SetColors(cols);
                    mesh.SetTriangles(tris, 0);
                    mesh.RecalculateNormals();
                    Persist(mesh, mesh.name);
                    go.AddComponent<MeshFilter>().sharedMesh = mesh;
                    go.AddComponent<MeshRenderer>().sharedMaterial = mat;
                    if (kSurfCollider) go.AddComponent<MeshCollider>().sharedMesh = mesh;
                    go.isStatic = true;
                    _surfTris += tris.Count / 3;
                }
            }
        }

        // 구워 온 노면 밝기 지도를 읽는다.
        //
        // 빌드에 들어가려면 에셋이어야 한다. 메모리에만 있는 텍스처는 씬에
        // 직렬화되기는 하지만 임포트 설정이 없어 압축도 밉맵도 없이 그대로
        // 들어가고, 3 백만 화소면 그것이 12 MB 다.
        static Texture2D LoadRoadTexture(RoadMap rm, string jsonDir)
        {
            if (!rm.valid || string.IsNullOrEmpty(rm.image)) return null;
            string src = Path.Combine(jsonDir, rm.image);
            if (!File.Exists(src)) { Debug.LogWarning("WorldVision: 노면 지도 없음 " + src); return null; }
            EnsureAssetDir();
            string dst = kAssetDir + "/road.png";

            // **그대로 복사하면 검은 바닥이 된다.**
            //
            // 카메라 노출은 하늘에 맞춰져 있어 아스팔트가 40/255 언저리로
            // 찍힌다. 그 값을 albedo 로 쓰면 조명이 다시 곱해져 거의 검게
            // 된다 - 실제 아스팔트의 반사율은 0.1~0.2 이지 0.02 가 아니다.
            //
            // 감마로 들어 올린다. 곱셈이 아니라 감마인 이유는, 곱하면 흰
            // 페인트가 먼저 포화해 차선이 노면에 녹아 버리기 때문이다.
            // 감마는 어두운 쪽을 더 많이 올리므로 대비가 남는다.
            {
                var raw = new Texture2D(2, 2, TextureFormat.RGBA32, false);
                raw.LoadImage(File.ReadAllBytes(src));
                var px = raw.GetPixels32();
                var lut = new byte[256];
                for (int i = 0; i < 256; ++i)
                {
                    lut[i] = (byte)Mathf.Clamp(
                        Mathf.RoundToInt(255f * Mathf.Pow(i / 255f, 0.52f)), 0, 255);
                }
                for (int i = 0; i < px.Length; ++i)
                {
                    byte g = lut[px[i].r];
                    px[i] = new Color32(g, g, (byte)Mathf.Min(255, g + 3), 255);
                }
                raw.SetPixels32(px);
                raw.Apply();
                File.WriteAllBytes(dst, raw.EncodeToPNG());
                UnityEngine.Object.DestroyImmediate(raw);
            }
            AssetDatabase.ImportAsset(dst, ImportAssetOptions.ForceUpdate);
            var imp = AssetImporter.GetAtPath(dst) as TextureImporter;
            if (imp != null)
            {
                imp.textureType = TextureImporterType.Default;
                imp.wrapMode = TextureWrapMode.Clamp;   // 밖으로 새면 길이 반복된다
                imp.filterMode = FilterMode.Bilinear;
                // **밉맵과 POT 보정을 끈다.**
                //
                // 이 그림은 2225x3998 처럼 2 의 거듭제곱이 아니다. 그대로 두면
                // 임포터가 POT 로 늘리고 밉맵까지 만드는데, 그렇게 나온 빌드가
                // 실행할 때 "level0 is corrupted / Position out of bounds" 로
                // 죽었다. PNG 를 치우면 멀쩡히 떴으므로 원인은 여기다.
                //
                // 밉맵은 어차피 필요 없다. 노면은 늘 비스듬히 가까이서 보므로
                // 축소본이 쓰일 일이 거의 없고, 밉맵이 섞이면 차선이 흐려진다.
                imp.npotScale = TextureImporterNPOTScale.None;
                imp.mipmapEnabled = false;
                imp.maxTextureSize = 2048;
                imp.textureCompression = TextureImporterCompression.Compressed;
                imp.SaveAndReimport();
            }
            return AssetDatabase.LoadAssetAtPath<Texture2D>(dst);
        }

        static bool Build(string path)
        {
            string text = File.ReadAllText(path);
            // JsonUtility 는 "class" 필드를 못 받는다. 읽기 전에 이름만 바꾼다.
            text = text.Replace("\"class\":", "\"cls\":");
            SceneFile s = JsonUtility.FromJson<SceneFile>(text);
            if (s == null || s.format != "worldvision-scene/1")
            {
                Debug.LogError("WorldVision: 형식이 맞지 않는 파일이다 - " + path);
                return false;
            }

            var root = new GameObject("WorldVision " + s.sequence + " @" + s.frame);
            Undo.RegisterCreatedObjectUndo(root, "Import WorldVision scene");

            // 자차가 섰던 자리를 표시로 남긴다. 사람을 놓을 때 쓴다 - 장면의
            // 한가운데는 건물 안일 수도 있지만 여기는 실제로 지나간 자리다.
            var egoGo = new GameObject("Ego");
            egoGo.transform.SetParent(root.transform, false);
            if (s.ego != null && s.ego.Length >= 3) egoGo.transform.position = P(s.ego);

            var matWall  = MakeMaterial("wv_wall",  new Color(0.72f, 0.70f, 0.66f));
            var matRoof  = MakeMaterial("wv_roof",  new Color(0.45f, 0.28f, 0.24f));
            var matTrunk = MakeMaterial("wv_trunk", new Color(0.30f, 0.22f, 0.15f));
            var matLeaf  = MakeMaterial("wv_leaf",  new Color(0.24f, 0.52f, 0.22f));
            var matPole  = MakeMaterial("wv_pole",  new Color(0.45f, 0.45f, 0.48f));
            var matCar   = MakeMaterial("wv_car",   new Color(0.20f, 0.55f, 0.52f));

            float surfCell;
            var surfTiles = ReadSurfaces(text, out surfCell);
            var roadMap = ReadRoadMap(text);
            // **노면 텍스처는 쓰지 않는다.**
            //
            // 구운 PNG 를 에셋으로 들이면 빌드는 성공하는데 실행할 때
            // "level0 is corrupted" 로 죽었다. PNG 를 치우면 멀쩡히 떴으므로
            // 원인은 그 임포트다 (2225x3998 은 2 의 거듭제곱이 아니다).
            // NPOT 보정과 밉맵을 꺼도 큰 장면에서는 여전히 죽었다.
            //
            // 그리고 이제 노면의 밝기는 점군이 나른다 - LiDAR 지도처럼
            // 점마다 색을 싣는 쪽이 원래 보여 주려던 것에 더 가깝다. PNG 는
            // 계속 내보낸다. 데이터이고, 언리얼 쪽은 그것을 쓴다.
            Texture2D roadTex = null;
            var gRoad = new GameObject("Surfaces").transform;
            gRoad.SetParent(root.transform, false);
            BuildSurfaces(gRoad, surfTiles, surfCell, roadMap, roadTex,
                          Path.GetDirectoryName(path));

            List<string> laneKinds; List<float> laneWidths;
            var lanes = ReadLanes(text, out laneKinds, out laneWidths);
            var gLane = new GameObject("Lanes").transform;
            gLane.SetParent(root.transform, false);
            BuildLanes(gLane, lanes, laneKinds, laneWidths);

            // 달릴 자리와 셀 숫자를 씬에 실어 둔다. 빌드된 실행 파일에는
            // JSON 이 없으므로, 런타임이 알아야 할 것은 여기서 넘겨야 한다.
            var routeGo = new GameObject("Route");
            routeGo.transform.SetParent(root.transform, false);
            var route = routeGo.AddComponent<Route>();
            route.points = ReadTrajectory(text).ToArray();

            var stat = routeGo.AddComponent<Stats>();
            stat.sequence = s.sequence;
            stat.frame = s.frame;
            stat.buildings = (s.buildings ?? new Building[0]).Length;
            stat.trees = (s.trees ?? new Tree[0]).Length;
            stat.poles = (s.poles ?? new Pole[0]).Length;
            stat.vehicles = (s.vehicles ?? new Vehicle[0]).Length;
            stat.lanes = lanes.Count;
            stat.surfaceCell = surfCell;
            stat.tRoad = surfTiles.FindAll(t => t.cls == 0).Count;
            stat.tSidewalk = surfTiles.FindAll(t => t.cls == 1).Count;
            stat.tGrass = surfTiles.FindAll(t => t.cls == 2).Count;
            stat.tOther = surfTiles.FindAll(t => t.cls == 3).Count;
            stat.roadMap = roadTex != null
                ? roadMap.width + "x" + roadMap.height + " px @ " + roadMap.cell + " m"
                : "";

            var gBuild = new GameObject("Buildings").transform;
            gBuild.SetParent(root.transform, false);
            foreach (var b in s.buildings ?? new Building[0])
            {
                Quaternion rot = Facing(b.forward);
                Vector3 c = P(b.center);
                var go = Box(gBuild, "Building", c, rot,
                             new Vector3(b.width, b.height, b.length), matWall);
                // range 는 이 집을 가장 가까이서 본 거리다. 신뢰도를 씬에
                // 남겨 두면 나중에 걸러 쓸 수 있다.
                go.isStatic = true;
                if (b.roof <= 0.0f) continue;
                // 박공은 큐브를 45 도로 눕혀 얹는다. 프리즘 메시를 따로
                // 만들 수도 있지만, 기본 도형만 쓰면 이 스크립트 하나로
                // 어떤 프로젝트에서든 돌아간다.
                float rise = b.roof;
                Vector3 up = Vector3.up * (b.height * 0.5f);
                var roof = Box(gBuild, "Roof", c + up + Vector3.up * (rise * 0.5f),
                               rot * Quaternion.Euler(0f, 0f, 45f),
                               new Vector3(rise * 1.42f, rise * 1.42f, b.length),
                               matRoof);
                roof.isStatic = true;
            }

            var gTree = new GameObject("Trees").transform;
            gTree.SetParent(root.transform, false);
            foreach (var t in s.trees ?? new Tree[0])
            {
                Vector3 foot = P(t.foot);
                float ry = Mathf.Clamp(t.height * 0.35f, 0.30f, 2.8f);
                float rr = Mathf.Max(t.canopy, ry * 0.70f);
                float cz = Mathf.Min(t.height * 0.58f, t.height - ry * 0.6f);
                float th = Mathf.Max(0.2f, cz - ry * 0.55f);

                var trunk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                trunk.name = "Trunk";
                trunk.transform.SetParent(gTree, false);
                trunk.transform.position = foot + Vector3.up * (th * 0.5f);
                // Unity 의 Cylinder 는 높이 2 가 기본이라 절반을 스케일로 준다.
                trunk.transform.localScale =
                    new Vector3(rr * 0.28f, th * 0.5f, rr * 0.28f);
                trunk.GetComponent<Renderer>().sharedMaterial = matTrunk;

                var leaf = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                leaf.name = "Canopy";
                leaf.transform.SetParent(gTree, false);
                leaf.transform.position = foot + Vector3.up * cz;
                leaf.transform.localScale = new Vector3(rr * 2f, ry * 2f, rr * 2f);
                leaf.GetComponent<Renderer>().sharedMaterial = matLeaf;
            }

            var gPole = new GameObject("Poles").transform;
            gPole.SetParent(root.transform, false);
            foreach (var p in s.poles ?? new Pole[0])
            {
                var go = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                go.name = "Pole";
                go.transform.SetParent(gPole, false);
                go.transform.position = P(p.foot) + Vector3.up * (p.height * 0.5f);
                go.transform.localScale = new Vector3(0.12f, p.height * 0.5f, 0.12f);
                go.GetComponent<Renderer>().sharedMaterial = matPole;
            }

            var gCar = new GameObject("Vehicles").transform;
            gCar.SetParent(root.transform, false);
            foreach (var v in s.vehicles ?? new Vehicle[0])
            {
                // size 는 (x, y, z) 가 폭·높이·길이다. 가장 긴 변을 길이로
                // 보는 것은 내보내는 쪽과 같은 규칙이다.
                float L = Mathf.Max(0.6f, Mathf.Max(v.size[0],
                                    Mathf.Max(v.size[1], v.size[2])));
                float Wd = Mathf.Max(0.4f, Mathf.Min(v.size[0], v.size[2]));
                float H = Mathf.Max(0.4f, v.size[1]);
                var go = Box(gCar, v.cls + (v.moving ? " (moving)" : ""),
                             P(v.center), Facing(v.forward),
                             new Vector3(Wd, H, L), matCar);
                go.isStatic = !v.moving;
            }

            Debug.Log(string.Format(
                "WorldVision: {0} 프레임 {1} - 건물 {2}, 나무 {3}, 기둥 {4}, 차량 {5}, "
                + "지표면 {6} 타일 (도로 {7} 인도 {8} 잔디 {9} 기타 {10}), 노면지도 {11}, 차선 {12}",
                s.sequence, s.frame,
                (s.buildings ?? new Building[0]).Length,
                (s.trees ?? new Tree[0]).Length,
                (s.poles ?? new Pole[0]).Length,
                (s.vehicles ?? new Vehicle[0]).Length,
                surfTiles.Count,
                surfTiles.FindAll(t => t.cls == 0).Count,
                surfTiles.FindAll(t => t.cls == 1).Count,
                surfTiles.FindAll(t => t.cls == 2).Count,
                surfTiles.FindAll(t => t.cls == 3).Count,
                roadTex != null ? roadMap.width + "x" + roadMap.height : "없음",
                lanes.Count));
            AssetDatabase.SaveAssets();
            Selection.activeGameObject = root;
            return true;
        }
    }
}
