// 지도를 **실행할 때** 짓는다.
//
// 처음에는 에디터에서 씬에 전부 세우고 그 씬을 빌드했다. 그런데 장면이 커지면
// 빌드는 성공하는데 실행할 때 "level0 is corrupted / Position out of bounds"
// 로 죽었다 - 작은 장면은 뜨고 큰 장면은 안 뜨는 식이라 원인을 좁히는 데
// 오래 걸렸다. 메시를 쪼개도, 텍스처를 빼도, 충돌체를 빼도 큰 장면은 죽었다.
//
// 그래서 데이터를 씬에서 꺼낸다. StreamingAssets 는 직렬화되지 않고 파일
// 그대로 복사되므로 씬은 텅 비어 있고, 지도는 실행할 때 읽어서 짓는다.
// 부수 효과가 더 크다: 데이터만 바꿔 넣으면 다시 빌드하지 않아도 된다.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace WorldVision
{
    public class Boot : MonoBehaviour
    {
        public string sceneName = "kitti_00";

        [NonSerialized] public Log log;
        [NonSerialized] public Stats stats;
        [NonSerialized] public Route route;
        [NonSerialized] public List<Detection> detections = new List<Detection>();
        [NonSerialized] public List<Lane> lanes = new List<Lane>();

        // 예측한 차선 한 줄.
        public class Lane
        {
            public List<Vector3> pts = new List<Vector3>();
            public bool center;
        }

        // 인식한 물체 하나. 화면에 상자와 이름표로 뜬다.
        public class Detection
        {
            public Vector3 center, size, forward;
            public string cls;
            public bool moving;
            public int seen;
        }

        void Awake()
        {
            log = gameObject.AddComponent<Log>();
            stats = gameObject.AddComponent<Stats>();
            route = gameObject.AddComponent<Route>();

            string dir = Application.streamingAssetsPath;
            log.Add("boot", "WorldVision-SLAM 실행", LogKind.Info);
            log.Add("boot", "데이터: " + dir, LogKind.Info);

            string json = Path.Combine(dir, sceneName + ".json");
            if (!File.Exists(json))
            {
                log.Add("boot", "장면 파일이 없다: " + json, LogKind.Error);
                return;
            }
            string text = File.ReadAllText(json);
            BuildFromJson(text);
            BuildCloud(Path.Combine(dir, sceneName + ".wvpc"));
            log.Add("boot", "지도 준비 완료", LogKind.Ok);
        }

        // ------------------------------------------------------------------
        // 숫자 읽기
        // ------------------------------------------------------------------
        // JsonUtility 를 안 쓴다. 중첩 배열을 못 읽고, 여기서 필요한 것은
        // 대부분 중첩 배열이다. 필요한 만큼만 손으로 훑는 편이 짧다.
        static float F(string s)
        {
            float v;
            float.TryParse(s.Trim(), NumberStyles.Float,
                           CultureInfo.InvariantCulture, out v);
            return v;
        }

        // [x, y, z] 를 유니티 좌표로. 오른손 +y up -> 왼손 +y up 이라 z 를 뒤집는다.
        static Vector3 V(string body)
        {
            string[] p = body.Split(',');
            if (p.Length < 3) return Vector3.zero;
            return new Vector3(F(p[0]), F(p[1]), -F(p[2]));
        }

        static List<string> Arrays(string text, string key, out int endAt)
        {
            var outp = new List<string>();
            endAt = -1;
            int at = text.IndexOf("\"" + key + "\"", StringComparison.Ordinal);
            if (at < 0) return outp;
            int open = text.IndexOf('[', at);
            if (open < 0) return outp;
            int depth = 0, i = open;
            for (; i < text.Length; ++i)
            {
                if (text[i] == '[') { if (++depth == 2) { int c = text.IndexOf(']', i); if (c < 0) break; outp.Add(text.Substring(i + 1, c - i - 1)); i = c; depth--; } }
                else if (text[i] == ']') { if (--depth <= 0) { endAt = i; break; } }
            }
            return outp;
        }

        static float Num(string text, string key, int from, float dflt)
        {
            int at = text.IndexOf("\"" + key + "\"", from, StringComparison.Ordinal);
            if (at < 0) return dflt;
            int c0 = text.IndexOf(':', at) + 1;
            int c1 = text.IndexOfAny(new[] { ',', '}' }, c0);
            if (c0 <= 0 || c1 < 0) return dflt;
            return F(text.Substring(c0, c1 - c0));
        }

        static string Str(string text, string key, int from)
        {
            int at = text.IndexOf("\"" + key + "\"", from, StringComparison.Ordinal);
            if (at < 0) return "";
            int q0 = text.IndexOf('"', text.IndexOf(':', at)) + 1;
            int q1 = text.IndexOf('"', q0);
            if (q0 <= 0 || q1 < 0) return "";
            return text.Substring(q0, q1 - q0);
        }

        // ------------------------------------------------------------------
        // 장면
        // ------------------------------------------------------------------
        void BuildFromJson(string text)
        {
            stats.sequence = Str(text, "sequence", 0);
            stats.frame = (int)Num(text, "frame", 0, 0);
            log.Add("scene", string.Format("{0}, 누적 {1} 프레임",
                stats.sequence, stats.frame), LogKind.Info);

            int e;
            // 궤적
            var traj = Arrays(text, "trajectory", out e);
            var pts = new List<Vector3>(traj.Count);
            foreach (var t in traj) pts.Add(V(t));
            route.points = pts.ToArray();
            log.Add("route", string.Format("궤적 {0} 점, {1:0} m",
                pts.Count, route.Length), LogKind.Ok);

            // 차량 - 상자와 이름표로 세운다
            int at = text.IndexOf("\"vehicles\"", StringComparison.Ordinal);
            if (at >= 0)
            {
                int i = at;
                while (true)
                {
                    int o = text.IndexOf("{\"class\"", i, StringComparison.Ordinal);
                    if (o < 0) o = text.IndexOf("{ \"class\"", i, StringComparison.Ordinal);
                    if (o < 0) break;
                    int close = text.IndexOf('}', o);
                    if (close < 0) break;
                    string body = text.Substring(o, close - o + 1);
                    var d = new Detection();
                    d.cls = Str(body, "class", 0);
                    d.seen = (int)Num(body, "seen", 0, 0);
                    d.moving = body.Contains("\"moving\": true");
                    int ce;
                    var cc = Arrays("{\"center\": [" + Between(body, "\"center\"") + "]}", "center", out ce);
                    d.center = V(Between(body, "\"center\""));
                    d.forward = V(Between(body, "\"forward\""));
                    var sz = Between(body, "\"size\"").Split(',');
                    if (sz.Length >= 3)
                        d.size = new Vector3(F(sz[0]), F(sz[1]), F(sz[2]));
                    detections.Add(d);
                    i = close + 1;
                    if (text.IndexOf('}', i) < 0) break;
                    int nxt = text.IndexOf("\"class\"", i, StringComparison.Ordinal);
                    int arrEnd = text.IndexOf("],", i, StringComparison.Ordinal);
                    if (nxt < 0 || (arrEnd >= 0 && arrEnd < nxt)) break;
                }
            }
            stats.vehicles = detections.Count;
            log.Add("detect", string.Format("차량 {0} 대 (움직임 {1})",
                detections.Count, detections.FindAll(d => d.moving).Count),
                LogKind.Ok);

            var bs = CountObjects(text, "buildings");
            var ts = CountObjects(text, "trees");
            var ps = CountObjects(text, "poles");
            stats.buildings = bs; stats.trees = ts; stats.poles = ps;
            log.Add("detect", string.Format("건물 {0} · 나무 {1} · 기둥 {2}",
                bs, ts, ps), LogKind.Ok);

            var surf = Arrays(text, "surfaces", out e);
            int[] cnt = new int[4];
            foreach (var t in surf)
            {
                string[] p = t.Split(',');
                if (p.Length < 5) continue;
                int c = Mathf.Clamp((int)F(p[4]), 0, 3);
                cnt[c]++;
            }
            stats.tRoad = cnt[0]; stats.tSidewalk = cnt[1];
            stats.tGrass = cnt[2]; stats.tOther = cnt[3];
            stats.surfaceCell = Num(text, "cell", text.IndexOf("\"surfaces\"",
                StringComparison.Ordinal), 0.5f);
            float a = stats.surfaceCell * stats.surfaceCell;
            log.Add("surface", string.Format(
                "도로 {0:0} m² · 인도 {1:0} m² · 잔디 {2:0} m²",
                cnt[0] * a, cnt[1] * a, cnt[2] * a), LogKind.Ok);

            ReadLanes(text);
            stats.lanes = lanes.Count;
            log.Add("lane", string.Format("차선 {0} 줄 (도로 구조에서 예측)",
                lanes.Count), lanes.Count > 0 ? LogKind.Ok : LogKind.Warn);
        }

        // 차선. {"kind": ..., "points": [[x,y,z], ...]} 가 여러 개다.
        void ReadLanes(string text)
        {
            int at = text.IndexOf("\"lanes\"", StringComparison.Ordinal);
            if (at < 0) return;
            int i = at;
            while (true)
            {
                int k0 = text.IndexOf("\"kind\"", i, StringComparison.Ordinal);
                if (k0 < 0) break;
                var L = new Lane();
                L.center = Str(text, "kind", k0) == "center";
                int p0 = text.IndexOf("\"points\"", k0, StringComparison.Ordinal);
                if (p0 < 0) break;
                int open = text.IndexOf('[', p0);
                int close = text.IndexOf("]]", open);
                if (open < 0 || close < 0) break;
                int j = open + 1;
                while (j < close + 1)
                {
                    int a2 = text.IndexOf('[', j);
                    if (a2 < 0 || a2 > close) break;
                    int b2 = text.IndexOf(']', a2);
                    if (b2 < 0) break;
                    L.pts.Add(V(text.Substring(a2 + 1, b2 - a2 - 1)));
                    j = b2 + 1;
                }
                if (L.pts.Count >= 2) lanes.Add(L);
                i = close + 2;
                int nxt = text.IndexOf("\"kind\"", i, StringComparison.Ordinal);
                int end = text.IndexOf("],", i, StringComparison.Ordinal);
                if (nxt < 0 || (end >= 0 && end < nxt)) break;
            }
        }

        static string Between(string body, string key)
        {
            int at = body.IndexOf(key, StringComparison.Ordinal);
            if (at < 0) return "";
            int o = body.IndexOf('[', at), c = body.IndexOf(']', o);
            if (o < 0 || c < 0) return "";
            return body.Substring(o + 1, c - o - 1);
        }

        static int CountObjects(string text, string key)
        {
            int at = text.IndexOf("\"" + key + "\"", StringComparison.Ordinal);
            if (at < 0) return 0;
            int open = text.IndexOf('[', at);
            if (open < 0) return 0;
            int n = 0, depth = 0;
            for (int i = open; i < text.Length; ++i)
            {
                char ch = text[i];
                if (ch == '[') depth++;
                else if (ch == ']') { if (--depth <= 0) break; }
                else if (ch == '{' && depth == 1) n++;
            }
            return n;
        }

        // ------------------------------------------------------------------
        // 점군
        // ------------------------------------------------------------------
        // 83 만 점을 6 만 점짜리 메시로 쪼개 올린다. 한 덩이로 올리면 인덱스가
        // UInt32 로 넘어가고, 그렇게 만든 빌드가 실행 중에 죽는 것을 이미
        // 보았다. 쪼개면 절두체 컬링도 같이 얻는다.
        public int cloudPoints;

        // 색을 무엇으로 칠할 것인가.
        //
        // 하나로 고정하면 안 된다. 높이 색은 구조를 보여 주고, 밝기 색은
        // 차선과 표지를 보여 주며, 클래스 색은 분류가 맞았는지를 보여 준다 -
        // 셋은 서로 다른 질문에 답하므로 바꿔 볼 수 있어야 한다.
        public enum Paint { Height, Intensity, Class }
        public Paint paint = Paint.Height;

        // 다시 칠하려면 원래 값이 있어야 한다. 색만 들고 있으면 되돌릴 수 없다.
        readonly List<Mesh> cloudMeshes = new List<Mesh>();
        readonly List<float[]> cloudRel = new List<float[]>();
        readonly List<byte[]> cloudInten = new List<byte[]>();
        readonly List<byte[]> cloudCls = new List<byte[]>();

        // **클래스마다 메시를 따로 만든다.**
        //
        // 켜고 끄려면 그래야 한다. 한 덩이에 섞어 두면 특정 클래스만 숨기는
        // 방법이 없다 - 색을 투명하게 해도 불투명 셰이더라 그대로 보이고,
        // 인덱스를 다시 만들면 83 만 점을 매번 다시 올려야 한다.
        //
        // 무엇을 숨길 수 있어야 하는가는 분명하다: 건물을 끄면 그 뒤의 길이
        // 보이고, 지면을 끄면 구조물만 남는다. 분류가 맞았는지도 그렇게 봐야
        // 드러난다 - 나무만 켜 놓고 그것이 정말 나무 자리인지 보는 식이다.
        public const int kClassN = 7;
        public static readonly string[] kClassName = {
            "미상", "지면", "건물", "담장", "나무", "기둥", "잔디" };
        public readonly List<Renderer>[] byClass = new List<Renderer>[kClassN];
        public readonly int[] classCount = new int[kClassN];
        public readonly bool[] classOn = new bool[kClassN];

        public void ShowClass(int c, bool on)
        {
            if (c < 0 || c >= kClassN || byClass[c] == null) return;
            classOn[c] = on;
            foreach (var r in byClass[c]) if (r != null) r.enabled = on;
            log.Add("view", string.Format("{0} {1}", kClassName[c],
                on ? "표시" : "숨김"), LogKind.Info);
        }

        // 투과 보기.
        //
        // 점을 더하기로 섞고 깊이를 쓰지 않으면 앞의 면이 뒤를 가리지 않아
        // 구조가 겹쳐 보인다 - 건물 안의 기둥, 나무 뒤의 담장이 드러난다.
        // **실제로 투과한 것이 아니다.** 가려진 것을 새로 관측한 것이 아니라
        // 이미 지도에 있던 점을 겹쳐 보이게 한 것뿐이고, 그래서 이 모드에서는
        // 무엇이 앞인지 알 수 없다.
        Material pointMat;

        public void SetXray(bool on)
        {
            if (pointMat == null) return;
            pointMat.SetInt("_SrcBlend", (int)(on
                ? UnityEngine.Rendering.BlendMode.One
                : UnityEngine.Rendering.BlendMode.One));
            pointMat.SetInt("_DstBlend", (int)(on
                ? UnityEngine.Rendering.BlendMode.One
                : UnityEngine.Rendering.BlendMode.Zero));
            pointMat.SetInt("_ZWrite", on ? 0 : 1);
            pointMat.renderQueue = on ? 3000 : 2000;
            log.Add("view", on ? "투과 보기 켜짐 (겹쳐 보이게 한 것이지 투과가 아니다)"
                               : "투과 보기 꺼짐", LogKind.Info);
        }

        public void Repaint(Paint p)
        {
            paint = p;
            for (int c = 0; c < cloudMeshes.Count; ++c)
            {
                var rel = cloudRel[c];
                var inten = cloudInten[c];
                var cls = cloudCls[c];
                var cols = new Color32[rel.Length];
                for (int i = 0; i < rel.Length; ++i)
                    cols[i] = Shade(p, rel[i], inten[i], cls[i]);
                cloudMeshes[c].colors32 = cols;
            }
            log.Add("paint", p == Paint.Height ? "색: 지면 위 높이"
                : p == Paint.Intensity ? "색: 관측 밝기" : "색: 분류",
                LogKind.Info);
        }

        public static Color32 Shade(Paint p, float rel, byte inten, byte cls)
        {
            if (p == Paint.Height) return Turbo(rel, inten);
            if (p == Paint.Intensity)
            {
                // 밝기만. 차선과 표지가 이때 가장 잘 보인다.
                float g = Mathf.Clamp01(inten / 255f);
                g = Mathf.Pow(g, 0.65f);
                return new Color32((byte)(g * 245f), (byte)(g * 250f),
                                   (byte)(g * 255f), 255);
            }
            // 분류. 뷰어가 화면에서 쓰는 색과 같게 둔다 - 다르면 두 화면을
            // 나란히 놓고 비교할 수가 없다.
            float k = 0.55f + 0.60f * (inten / 255f);
            Color b2;
            switch (cls)
            {
                case 1:  b2 = new Color(0.42f, 0.42f, 0.44f); break;  // 지면
                case 2:  b2 = new Color(0.55f, 0.68f, 0.86f); break;  // 건물
                case 3:  b2 = new Color(0.80f, 0.72f, 0.45f); break;  // 담장
                case 4:  b2 = new Color(0.34f, 0.80f, 0.38f); break;  // 나무
                case 5:  b2 = new Color(0.95f, 0.55f, 0.85f); break;  // 기둥
                case 6:  b2 = new Color(0.55f, 0.80f, 0.35f); break;  // 잔디
                default: b2 = new Color(0.45f, 0.45f, 0.50f); break;  // 미상
            }
            return new Color32(
                (byte)Mathf.Clamp(b2.r * k * 255f, 0f, 255f),
                (byte)Mathf.Clamp(b2.g * k * 255f, 0f, 255f),
                (byte)Mathf.Clamp(b2.b * k * 255f, 0f, 255f), 255);
        }

        void BuildCloud(string path)
        {
            if (!File.Exists(path))
            {
                log.Add("cloud", "점군 파일이 없다: " + path, LogKind.Warn);
                return;
            }
            byte[] raw = File.ReadAllBytes(path);
            if (raw.Length < 24 || raw[0] != 'W' || raw[1] != 'V' ||
                raw[2] != 'P' || raw[3] != 'C')
            {
                log.Add("cloud", "점군 형식이 아니다", LogKind.Error);
                return;
            }
            int count = BitConverter.ToInt32(raw, 8);
            var org = new Vector3(BitConverter.ToSingle(raw, 12),
                                  BitConverter.ToSingle(raw, 16),
                                  -BitConverter.ToSingle(raw, 20));
            const int rec = 18;
            int need = 24 + count * rec;
            if (need > raw.Length)
            {
                log.Add("cloud", "점군이 잘렸다", LogKind.Error);
                return;
            }

            var sh = Shader.Find("WorldVision/Point");
            var mat = new Material(sh != null ? sh : Shader.Find("Unlit/Color"));
            mat.name = "wv_point";
            pointMat = mat;

            var parent = new GameObject("PointCloud").transform;
            const int kChunk = 60000;

            // 먼저 클래스별로 색인을 모은다. 점 하나를 두 번 읽지 않으려고
            // 위치와 속성도 이때 함께 뽑아 둔다.
            var idxOf = new List<int>[kClassN];
            for (int c = 0; c < kClassN; ++c) { idxOf[c] = new List<int>(); classOn[c] = true; }
            var pos = new Vector3[count];
            var rel = new float[count];
            var inten = new byte[count];
            var cls = new byte[count];
            for (int i = 0; i < count; ++i)
            {
                int o = 24 + i * rec;
                pos[i] = new Vector3(BitConverter.ToSingle(raw, o),
                                     BitConverter.ToSingle(raw, o + 4),
                                     -BitConverter.ToSingle(raw, o + 8)) + org;
                rel[i] = BitConverter.ToSingle(raw, o + 12);
                inten[i] = raw[o + 16];
                byte c2 = raw[o + 17];
                if (c2 >= kClassN) c2 = 0;
                cls[i] = c2;
                idxOf[c2].Add(i);
                classCount[c2]++;
            }

            int made = 0, meshes = 0;
            for (int c = 0; c < kClassN; ++c)
            {
                byClass[c] = new List<Renderer>();
                var list = idxOf[c];
                if (list.Count == 0) continue;
                int chunks = (list.Count + kChunk - 1) / kChunk;
                for (int ch = 0; ch < chunks; ++ch)
                {
                    int from = ch * kChunk;
                    int n = Mathf.Min(kChunk, list.Count - from);
                    var verts = new Vector3[n];
                    var cols = new Color32[n];
                    var idx = new int[n];
                    var relBuf = new float[n];
                    var intenBuf = new byte[n];
                    var clsBuf = new byte[n];
                    for (int i = 0; i < n; ++i)
                    {
                        int src = list[from + i];
                        verts[i] = pos[src];
                        relBuf[i] = rel[src];
                        intenBuf[i] = inten[src];
                        clsBuf[i] = cls[src];
                        cols[i] = Shade(paint, rel[src], inten[src], cls[src]);
                        idx[i] = i;
                    }
                    var mesh = new Mesh();
                    mesh.name = "cloud_" + kClassName[c] + ch;
                    mesh.vertices = verts;
                    mesh.colors32 = cols;
                    mesh.SetIndices(idx, MeshTopology.Points, 0);
                    // 점은 경계를 스스로 못 정한다 - 재계산해야 컬링이 맞는다.
                    mesh.RecalculateBounds();
                    var go = new GameObject(kClassName[c] + " " + ch);
                    go.transform.SetParent(parent, false);
                    go.AddComponent<MeshFilter>().sharedMesh = mesh;
                    var mr = go.AddComponent<MeshRenderer>();
                    mr.sharedMaterial = mat;
                    mr.shadowCastingMode =
                        UnityEngine.Rendering.ShadowCastingMode.Off;
                    mr.receiveShadows = false;
                    byClass[c].Add(mr);
                    cloudMeshes.Add(mesh);
                    cloudRel.Add(relBuf);
                    cloudInten.Add(intenBuf);
                    cloudCls.Add(clsBuf);
                    made += n;
                    meshes++;
                }
            }
            cloudPoints = made;
            log.Add("cloud", string.Format("점군 {0:n0} 점, 메시 {1}",
                made, meshes), LogKind.Ok);
            log.Add("cloud", string.Format(
                "지면 {0:n0} · 건물 {1:n0} · 나무 {2:n0} · 잔디 {3:n0} · 담장 {4:n0} · 미상 {5:n0}",
                classCount[1], classCount[2], classCount[4], classCount[6],
                classCount[3], classCount[0]), LogKind.Info);
        }

        // 높이 색. LiDAR 지도의 그 색이다 - 노면이 파랗고 지붕이 붉다.
        //
        // 밝기를 곱해 둔다. 높이만 쓰면 같은 높이의 아스팔트와 흰 페인트가
        // 같은 색이 되어 차선이 사라진다.
        public static Color32 Turbo(float h, byte inten)
        {
            float t = Mathf.Clamp01(h / 12f);
            float r, g, b;
            if (t < 0.25f) { float u = t / 0.25f; r = 0.10f; g = 0.20f + 0.60f * u; b = 0.85f + 0.10f * u; }
            else if (t < 0.5f) { float u = (t - 0.25f) / 0.25f; r = 0.10f + 0.10f * u; g = 0.80f + 0.15f * u; b = 0.95f - 0.75f * u; }
            else if (t < 0.75f) { float u = (t - 0.5f) / 0.25f; r = 0.20f + 0.70f * u; g = 0.95f - 0.20f * u; b = 0.20f - 0.15f * u; }
            else { float u = (t - 0.75f) / 0.25f; r = 0.90f + 0.08f * u; g = 0.75f - 0.60f * u; b = 0.05f; }
            float k = 0.55f + 0.75f * (inten / 255f);
            return new Color32(
                (byte)Mathf.Clamp(r * k * 255f, 0f, 255f),
                (byte)Mathf.Clamp(g * k * 255f, 0f, 255f),
                (byte)Mathf.Clamp(b * k * 255f, 0f, 255f), 255);
        }

        // ------------------------------------------------------------------
        // 카메라 프레임
        // ------------------------------------------------------------------
        // 지도의 근거가 된 그 화면이다. 3D 만 보여 주면 그럴듯해 보이는 것과
        // 실제로 본 것을 구분할 길이 없다 - 옆에 원본을 두면 바로 갈린다.
        //
        // 통째로 들고 있지 않는다. 400 장을 다 열면 몇백 MB 이므로, 보이는
        // 것만 그때 읽고 최근 몇 장만 남긴다.
        readonly Dictionary<int, Texture2D> camCache = new Dictionary<int, Texture2D>();
        readonly List<int> camOrder = new List<int>();
        string[] camFiles;

        public Texture2D CamFrame(float t)
        {
            if (camFiles == null)
            {
                string d = Path.Combine(Application.streamingAssetsPath, "cam");
                camFiles = Directory.Exists(d) ? Directory.GetFiles(d, "*.jpg") : new string[0];
                Array.Sort(camFiles);
                log.Add("cam", camFiles.Length > 0
                    ? string.Format("카메라 프레임 {0} 장", camFiles.Length)
                    : "카메라 프레임 없음",
                    camFiles.Length > 0 ? LogKind.Ok : LogKind.Warn);
            }
            if (camFiles.Length == 0) return null;
            int i = Mathf.Clamp(Mathf.RoundToInt(t * (camFiles.Length - 1)),
                                0, camFiles.Length - 1);
            Texture2D tex;
            if (camCache.TryGetValue(i, out tex)) return tex;
            tex = new Texture2D(2, 2, TextureFormat.RGB24, false);
            tex.LoadImage(File.ReadAllBytes(camFiles[i]));
            tex.Apply(false, true);
            camCache[i] = tex;
            camOrder.Add(i);
            if (camOrder.Count > 24)
            {
                int old = camOrder[0];
                camOrder.RemoveAt(0);
                if (camCache.ContainsKey(old))
                {
                    Destroy(camCache[old]);
                    camCache.Remove(old);
                }
            }
            return tex;
        }

    }

    public enum LogKind { Info, Ok, Warn, Error, Detect }

    // 아래쪽 터미널에 흐르는 줄들.
    //
    // 그림만 보여 주면 무엇을 근거로 그렸는지가 안 보인다. 인식이 일어난
    // 순간을 글로 남기면, 화면의 상자가 어디서 왔는지 따라갈 수 있다.
    public class Log : MonoBehaviour
    {
        public struct Line
        {
            public float t;
            public string tag, msg;
            public LogKind kind;
        }

        public readonly List<Line> lines = new List<Line>();

        public void Add(string tag, string msg, LogKind kind)
        {
            lines.Add(new Line { t = Time.realtimeSinceStartup, tag = tag,
                                 msg = msg, kind = kind });
            if (lines.Count > 400) lines.RemoveRange(0, 100);
        }
    }
}
