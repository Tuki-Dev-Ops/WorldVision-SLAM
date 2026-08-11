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
            return m;
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

            for (int c = 0; c < 4; ++c)
            {
                var verts = new List<Vector3>();
                var uvs = new List<Vector2>();
                var tris = new List<int>();
                foreach (var t in tiles)
                {
                    if (t.cls != c) continue;
                    int b = verts.Count;
                    verts.Add(t.p + new Vector3(-h, 0, -h));
                    verts.Add(t.p + new Vector3( h, 0, -h));
                    verts.Add(t.p + new Vector3( h, 0,  h));
                    verts.Add(t.p + new Vector3(-h, 0,  h));
                    for (int k = 0; k < 4; ++k)
                    {
                        // **UV 는 월드 좌표에서 뽑는다.** 타일마다 0..1 을 주면
                        // 차선 한 줄이 타일마다 끊겨 점선이 된다. 구워 온
                        // 지도는 하나의 큰 그림이므로 그 위의 자리를 그대로
                        // 가리켜야 무늬가 이어진다.
                        Vector3 d = verts[b + k] - rm.origin;
                        uvs.Add(rm.valid
                            ? new Vector2(Vector3.Dot(d, rm.axisU) / spanU,
                                          Vector3.Dot(d, rm.axisV) / spanV)
                            : new Vector2(0.5f, 0.5f));
                    }
                    tris.Add(b); tris.Add(b + 3); tris.Add(b + 2);
                    tris.Add(b); tris.Add(b + 2); tris.Add(b + 1);
                }
                if (verts.Count == 0) continue;

                var go = new GameObject(kSurfName[c]);
                go.transform.SetParent(parent, false);
                var mesh = new Mesh();
                mesh.indexFormat = verts.Count > 65000
                    ? UnityEngine.Rendering.IndexFormat.UInt32
                    : UnityEngine.Rendering.IndexFormat.UInt16;
                mesh.SetVertices(verts);
                mesh.SetUVs(0, uvs);
                mesh.SetTriangles(tris, 0);
                mesh.RecalculateNormals();
                go.AddComponent<MeshFilter>().sharedMesh = mesh;
                go.AddComponent<MeshRenderer>().sharedMaterial = mats[c];
                // 딛고 설 수 있어야 바닥이다.
                go.AddComponent<MeshCollider>().sharedMesh = mesh;
                go.isStatic = true;
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
            const string dstDir = "Assets/WorldVision";
            Directory.CreateDirectory(dstDir);
            string dst = dstDir + "/road.png";

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
                imp.mipmapEnabled = true;
                imp.maxTextureSize = 4096;              // 2472 px 를 줄이지 않는다
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
            var roadTex = LoadRoadTexture(roadMap, Path.GetDirectoryName(path));
            var gRoad = new GameObject("Surfaces").transform;
            gRoad.SetParent(root.transform, false);
            BuildSurfaces(gRoad, surfTiles, surfCell, roadMap, roadTex,
                          Path.GetDirectoryName(path));

            List<string> laneKinds; List<float> laneWidths;
            var lanes = ReadLanes(text, out laneKinds, out laneWidths);
            var gLane = new GameObject("Lanes").transform;
            gLane.SetParent(root.transform, false);
            BuildLanes(gLane, lanes, laneKinds, laneWidths);

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
            Selection.activeGameObject = root;
            return true;
        }
    }
}
