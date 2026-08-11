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

    // 노면 타일 하나. [x, y, z, 밝기]
    public struct RoadTile
    {
        public Vector3 p;
        public float v;     // 0..1
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

        [MenuItem("WorldVision/Import Scene (JSON)")]
        public static void Import()
        {
            string path = EditorUtility.OpenFilePanel(
                "WorldVision scene", Application.dataPath, "json");
            if (string.IsNullOrEmpty(path)) return;
            Build(path);
        }

        // **노면만 직접 파싱한다.**
        //
        // JsonUtility 는 중첩 배열(float[][])을 못 읽는다. 나머지는 전부
        // 객체 배열이라 문제가 없고, 노면만 [x,y,z,i] 형태라 여기서만
        // 숫자를 훑는다 - 이것 하나 때문에 JSON 라이브러리를 들이지 않는다.
        static List<RoadTile> ReadRoad(string text, out float cell)
        {
            var outp = new List<RoadTile>();
            cell = 1.0f;
            int at = text.IndexOf("\"road\"");
            if (at < 0) return outp;
            int cAt = text.IndexOf("\"cell\"", at);
            if (cAt >= 0)
            {
                int c0 = text.IndexOf(':', cAt) + 1;
                int c1 = text.IndexOfAny(new[] { ',', '}' }, c0);
                float.TryParse(text.Substring(c0, c1 - c0).Trim(),
                               System.Globalization.NumberStyles.Float,
                               System.Globalization.CultureInfo.InvariantCulture,
                               out cell);
            }
            int tAt = text.IndexOf('[', text.IndexOf("\"tiles\"", at));
            if (tAt < 0) return outp;
            int i = tAt + 1;
            var num = new float[4];
            while (i < text.Length)
            {
                int open = text.IndexOf('[', i);
                if (open < 0) break;
                int close = text.IndexOf(']', open);
                if (close < 0) break;
                string body = text.Substring(open + 1, close - open - 1);
                string[] parts = body.Split(',');
                if (parts.Length >= 4)
                {
                    bool ok = true;
                    for (int k = 0; k < 4; ++k)
                    {
                        ok &= float.TryParse(parts[k].Trim(),
                                System.Globalization.NumberStyles.Float,
                                System.Globalization.CultureInfo.InvariantCulture,
                                out num[k]);
                    }
                    if (ok)
                    {
                        outp.Add(new RoadTile {
                            p = new Vector3(num[0], num[1], -num[2]),
                            v = Mathf.Clamp01(num[3] / 255f) });
                    }
                }
                i = close + 1;
                int nxt = text.IndexOfAny(new[] { '[', ']' }, i);
                if (nxt < 0 || text[nxt] == ']') break;
            }
            return outp;
        }

        // 노면을 메시 하나로 만든다. 타일마다 GameObject 를 두면 수천 개가
        // 되어 씬이 무거워지고, 어차피 하나의 바닥이라 나눌 이유가 없다.
        // 밝기 단계. 마지막 칸이 차선이다 - 도료는 아스팔트보다 훨씬 밝게
        // 되돌아오므로 상위 구간만 흰색으로 뽑아 두면 선이 선으로 보인다.
        static readonly float[] kRoadStep = { 0.16f, 0.24f, 0.34f, 0.48f, 0.90f };

        static void BuildRoad(Transform parent, List<RoadTile> tiles, float cell)
        {
            if (tiles.Count == 0) return;

            // **정점 색이 아니라 서브메시로 나눈다.**
            //
            // 처음에는 SetColors 로 밝기를 실었는데 화면에는 흰 리본 하나만
            // 나왔다. Lit 셰이더는 COLOR_0 을 읽지 않으므로 밝기가 통째로
            // 버려졌던 것이다. 셰이더를 새로 쓰는 대신 단계별 서브메시로
            // 나누면 어떤 렌더 파이프라인에서도 그대로 보인다.
            int nb = kRoadStep.Length;
            var verts = new List<Vector3>(tiles.Count * 4);
            var tris = new List<int>[nb];
            for (int k = 0; k < nb; ++k) tris[k] = new List<int>();
            float h = cell * 0.5f;
            foreach (var t in tiles)
            {
                int b = verts.Count;
                verts.Add(t.p + new Vector3(-h, 0, -h));
                verts.Add(t.p + new Vector3( h, 0, -h));
                verts.Add(t.p + new Vector3( h, 0,  h));
                verts.Add(t.p + new Vector3(-h, 0,  h));
                // 밝기를 제곱해 대비를 세운다 - 뷰어가 화면에서 하는 것과 같다.
                int bucket = Mathf.Clamp((int)(t.v * t.v * nb), 0, nb - 1);
                var tl = tris[bucket];
                tl.Add(b); tl.Add(b + 3); tl.Add(b + 2);
                tl.Add(b); tl.Add(b + 2); tl.Add(b + 1);
            }

            var go = new GameObject("Road");
            go.transform.SetParent(parent, false);
            var mesh = new Mesh();
            mesh.indexFormat = verts.Count > 65000
                ? UnityEngine.Rendering.IndexFormat.UInt32
                : UnityEngine.Rendering.IndexFormat.UInt16;
            mesh.SetVertices(verts);
            mesh.subMeshCount = nb;
            var mats = new Material[nb];
            for (int k = 0; k < nb; ++k)
            {
                mesh.SetTriangles(tris[k], k);
                float g = kRoadStep[k];
                mats[k] = MakeMaterial("wv_road" + k, new Color(g, g, g * 1.02f));
            }
            mesh.RecalculateNormals();
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            go.AddComponent<MeshRenderer>().sharedMaterials = mats;
            // 달릴 수 있어야 바닥이다.
            go.AddComponent<MeshCollider>().sharedMesh = mesh;
            go.isStatic = true;
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

            var matWall  = MakeMaterial("wv_wall",  new Color(0.72f, 0.70f, 0.66f));
            var matRoof  = MakeMaterial("wv_roof",  new Color(0.45f, 0.28f, 0.24f));
            var matTrunk = MakeMaterial("wv_trunk", new Color(0.30f, 0.22f, 0.15f));
            var matLeaf  = MakeMaterial("wv_leaf",  new Color(0.24f, 0.52f, 0.22f));
            var matPole  = MakeMaterial("wv_pole",  new Color(0.45f, 0.45f, 0.48f));
            var matCar   = MakeMaterial("wv_car",   new Color(0.20f, 0.55f, 0.52f));

            float roadCell;
            var roadTiles = ReadRoad(text, out roadCell);
            var gRoad = new GameObject("Ground").transform;
            gRoad.SetParent(root.transform, false);
            BuildRoad(gRoad, roadTiles, roadCell);

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
                "WorldVision: {0} 프레임 {1} - 건물 {2}, 나무 {3}, 기둥 {4}, 차량 {5}, 노면 {6} 타일",
                s.sequence, s.frame,
                (s.buildings ?? new Building[0]).Length,
                (s.trees ?? new Tree[0]).Length,
                (s.poles ?? new Pole[0]).Length,
                (s.vehicles ?? new Vehicle[0]).Length,
                roadTiles.Count));
            Selection.activeGameObject = root;
            return true;
        }
    }
}
