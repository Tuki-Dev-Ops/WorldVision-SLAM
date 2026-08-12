// 세운 장면을 **돌려 본다.**
//
// 형상이 제자리에 있는지는 위에서 한 장 찍으면 알 수 있지만, 그것이 주행할 수
// 있는 세계인지는 달려 봐야 안다 - 노면이 이어져 있는지, 차선이 코너에서
// 끊기는지, 인도가 도로를 침범하는지는 정지 화면에서 보이지 않는다.
//
// 달릴 자리를 지어내지 않는다. 이 시퀀스에서 차가 실제로 지나간 자리가
// 궤적으로 들어와 있고, 그것이 곧 정답 주행이다. 속도까지 그 안에 있다.
using System.Collections.Generic;
using UnityEngine;

namespace WorldVision
{
    // 자차가 지나간 자리.
    public class Route : MonoBehaviour
    {
        public Vector3[] points;

        public float Length
        {
            get
            {
                float s = 0f;
                if (points == null) return 0f;
                for (int i = 0; i + 1 < points.Length; ++i)
                    s += Vector3.Distance(points[i], points[i + 1]);
                return s;
            }
        }

        // 거리 s (m) 위의 자리와 진행 방향.
        public bool Sample(float s, out Vector3 pos, out Vector3 fwd)
        {
            pos = Vector3.zero; fwd = Vector3.forward;
            if (points == null || points.Length < 2) return false;
            float acc = 0f;
            for (int i = 0; i + 1 < points.Length; ++i)
            {
                float seg = Vector3.Distance(points[i], points[i + 1]);
                if (seg < 1e-4f) continue;
                if (acc + seg >= s)
                {
                    float t = (s - acc) / seg;
                    pos = Vector3.Lerp(points[i], points[i + 1], t);
                    fwd = (points[i + 1] - points[i]).normalized;
                    return true;
                }
                acc += seg;
            }
            pos = points[points.Length - 1];
            fwd = (points[points.Length - 1] - points[points.Length - 2]).normalized;
            return true;
        }
    }

    // 무엇이 몇 개 인식되었는가.
    public class Stats : MonoBehaviour
    {
        public string sequence = "";
        public int frame;
        public int buildings, trees, poles, vehicles, lanes;
        public int tRoad, tSidewalk, tGrass, tOther;
        public float surfaceCell = 0.5f;
        public string roadMap = "";
    }

    // 뷰와 주행과 화면 표시를 한곳에서 맡는다.
    public class Director : MonoBehaviour
    {
        public enum View { Driver, Chase, Aerial, Free }

        public Route route;
        public Stats stats;
        public Camera cam;
        public Transform car;
        public Player free;
        public Boot boot;
        Renderer[] carRenderers;

        View view = View.Chase;
        bool playing = true;
        bool showLabels = true;
        bool showTerminal = true;
        float s;                    // 경로 위 거리 (m)
        float speed = 8f;           // m/s
        float routeLen = 1f;
        bool help = true;
        readonly List<float> frameMs = new List<float>();

        // 화면에 쓰는 색. 지도에 쓰는 것과 같아야 범례가 범례다.
        static readonly Color cRoad = new Color(0.62f, 0.62f, 0.64f);
        static readonly Color cSide = new Color(0.74f, 0.72f, 0.69f);
        static readonly Color cGrass = new Color(0.30f, 0.46f, 0.22f);
        static readonly Color cBuild = new Color(0.72f, 0.70f, 0.66f);
        static readonly Color cTree = new Color(0.24f, 0.52f, 0.22f);
        static readonly Color cCar = new Color(0.20f, 0.55f, 0.52f);
        static readonly Color cLane = new Color(0.94f, 0.88f, 0.45f);

        Texture2D panelTex, barTex;
        Bounds world;
        Vector2 panelScroll;
        float panelH = 1f;      // 그리면서 잰 왼쪽 패널의 내용 높이

        // 항공뷰 배율. 1 이면 장면 전체, 작을수록 가까이.
        //
        // 전체를 한 화면에 담는 것과 한 블록을 들여다보는 것은 다른 일이다.
        // 전체만 보이면 549 m 가 손바닥만 하게 들어와 아무 것도 못 읽는다.
        float aerialZoom = 1f;
        bool xray;

        // 화면 배치
        // ---------
        //   왼쪽        지표 · 범례 · 보기
        //   가운데      3D (점군 지도)
        //   아래        터미널 - 인식이 일어난 순간을 글로
        //   오른쪽 아래 카메라 - 근거가 된 그 화면
        //
        // 무엇을 봤고(카메라), 무엇으로 읽었고(3D), 왜 그렇게 읽었는가
        // (터미널)가 한 화면에 있어야 한다.
        const float kPanelW = 336f;
        const float kTermH = 176f;
        const float kCamW = 348f;

        void Start()
        {
            // 런타임에 지도를 짓는 구조라 Boot 가 만든 것을 여기서 받는다.
            if (boot != null)
            {
                if (route == null) route = boot.route;
                if (stats == null) stats = boot.stats;
            }
            if (route != null) routeLen = Mathf.Max(1f, route.Length);
            panelTex = Solid(new Color(0.045f, 0.055f, 0.075f, 0.94f));
            barTex = Solid(new Color(1f, 1f, 1f, 1f));
            world = WorldBounds();
            if (car != null) carRenderers = car.GetComponentsInChildren<Renderer>();
            Apply();
        }

        static Texture2D Solid(Color c)
        {
            var t = new Texture2D(1, 1);
            t.SetPixel(0, 0, c);
            t.Apply();
            return t;
        }

        static Bounds WorldBounds()
        {
            var b = new Bounds(Vector3.zero, Vector3.zero);
            bool got = false;
            foreach (var r in UnityEngine.Object.FindObjectsByType<Renderer>(
                         FindObjectsSortMode.None))
            {
                if (!got) { b = r.bounds; got = true; }
                else b.Encapsulate(r.bounds);
            }
            return b;
        }

        void Update()
        {
            frameMs.Add(Time.unscaledDeltaTime * 1000f);
            if (frameMs.Count > 120) frameMs.RemoveAt(0);

            if (Input.GetKeyDown(KeyCode.Alpha1)) { view = View.Driver; Apply(); }
            if (Input.GetKeyDown(KeyCode.Alpha2)) { view = View.Chase; Apply(); }
            if (Input.GetKeyDown(KeyCode.Alpha3)) { view = View.Aerial; Apply(); }
            if (Input.GetKeyDown(KeyCode.Alpha4)) { view = View.Free; Apply(); }
            if (Input.GetKeyDown(KeyCode.Space)) playing = !playing;
            if (Input.GetKeyDown(KeyCode.H)) help = !help;
            if (Input.GetKeyDown(KeyCode.L)) showLabels = !showLabels;
            if (Input.GetKeyDown(KeyCode.T)) showTerminal = !showTerminal;
            if (Input.GetKeyDown(KeyCode.X) && boot != null)
            {
                // X-ray. 점을 더하기로 섞고 깊이를 쓰지 않으면 앞뒤가 겹쳐
                // 보인다 - 건물 안의 기둥, 나무 뒤의 담장처럼 가려진 것이
                // 드러난다. 실제로 투과한 것이 아니라 **겹쳐 보이게** 한
                // 것이므로, 무엇이 앞인지는 이 모드에서 알 수 없다.
                xray = !xray;
                boot.SetXray(xray);
            }
            if (Input.GetKeyDown(KeyCode.C) && boot != null)
            {
                boot.Repaint((Boot.Paint)(((int)boot.paint + 1) % 3));
            }
            // 클래스 켜고 끄기. 건물을 끄면 그 뒤의 길이 보이고, 지면을 끄면
            // 구조물만 남는다 - 분류가 맞았는지는 그렇게 봐야 드러난다.
            if (boot != null)
            {
                for (int c = 0; c < Boot.kClassN; ++c)
                {
                    if (Input.GetKeyDown(KeyCode.F1 + c))
                        boot.ShowClass(c, !boot.classOn[c]);
                }
            }
            // 패널 위에서 휠을 굴리면 스크롤. 3D 위에서는 속도로 쓴다.
            float wheel = Input.mouseScrollDelta.y;
            if (Mathf.Abs(wheel) > 0.01f)
            {
                if (Input.mousePosition.x < kPanelW)
                    panelScroll.y = Mathf.Clamp(panelScroll.y - wheel * 40f,
                                                0f, Mathf.Max(0f, panelH - Screen.height));
                else if (view == View.Aerial)
                    // 항공뷰에서 휠은 줌이다. 지도를 보는 창에서 휠이 줌이
                    // 아니면 어디서 줌을 하겠는가.
                    aerialZoom = Mathf.Clamp(aerialZoom * (1f - wheel * 0.12f),
                                             0.04f, 1.6f);
                else
                    speed = Mathf.Clamp(speed + wheel * 2f, 1f, 40f);
            }
            if (Input.GetKeyDown(KeyCode.R)) s = 0f;
            if (Input.GetKeyDown(KeyCode.Q)) Application.Quit();
            if (Input.GetKey(KeyCode.LeftBracket))
                speed = Mathf.Max(1f, speed - 12f * Time.deltaTime);
            if (Input.GetKey(KeyCode.RightBracket))
                speed = Mathf.Min(40f, speed + 12f * Time.deltaTime);

            // 좌우로 경로를 훑는다. 그림에 쓸 구간을 찾을 때 재생을 기다릴
            // 이유가 없다.
            float scrub = 0f;
            if (Input.GetKey(KeyCode.LeftArrow)) scrub -= 1f;
            if (Input.GetKey(KeyCode.RightArrow)) scrub += 1f;
            if (Mathf.Abs(scrub) > 0f) s += scrub * 25f * Time.deltaTime;

            if (playing && view != View.Free) s += speed * Time.deltaTime;
            if (s > routeLen) s = 0f;
            if (s < 0f) s = routeLen;

            Drive();
            if (view != View.Free) Frame();
            Sense();
        }

        void Apply()
        {
            bool freeMode = view == View.Free;
            if (free != null)
            {
                free.enabled = freeMode;
                var cc = free.GetComponent<CharacterController>();
                if (cc != null) cc.enabled = freeMode;
                if (freeMode && car != null)
                    free.transform.position = car.position + Vector3.up * 2.5f;
            }
            // 운전석에서는 자차를 숨긴다. 안 숨기면 화면 절반이 자기 보닛이다.
            if (carRenderers != null)
            {
                bool show = view != View.Driver;
                foreach (var r in carRenderers) if (r != null) r.enabled = show;
            }
            if (cam != null)
            {
                cam.orthographic = view == View.Aerial;
                // 점군 지도는 검은 배경에서 읽는다. 하늘을 그리면 색이 죽는다.
                cam.clearFlags = CameraClearFlags.SolidColor;
                cam.backgroundColor = new Color(0.02f, 0.025f, 0.035f);
                if (freeMode)
                {
                    cam.transform.SetParent(free != null ? free.transform : null, false);
                    cam.transform.localPosition = new Vector3(0f, 1.62f, 0f);
                    cam.transform.localRotation = Quaternion.identity;
                }
                else
                {
                    cam.transform.SetParent(null, true);
                }
                Cursor.lockState = freeMode ? CursorLockMode.Locked : CursorLockMode.None;
                Cursor.visible = !freeMode;
            }
        }

        // **지금 자차 앞에 무엇이 들어왔는가.**
        //
        // 부팅 로그만 흐르면 터미널이 정지 화면이다. 달리면서 시야에 물체가
        // 들어오는 순간을 적으면, 화면의 상자가 어디서 왔는지 글로도 따라갈
        // 수 있다 - 그림과 로그가 같은 것을 가리켜야 한다.
        readonly HashSet<int> seenNear = new HashSet<int>();
        float senseAt;

        void Sense()
        {
            if (boot == null || car == null || boot.log == null) return;
            // 매 프레임 다 훑을 이유가 없다. 물체는 초 단위로 들어온다.
            if (Time.time - senseAt < 0.25f) return;
            senseAt = Time.time;

            var near = new HashSet<int>();
            for (int i = 0; i < boot.detections.Count; ++i)
            {
                var d = boot.detections[i];
                Vector3 v = d.center - car.position;
                float dist = v.magnitude;
                if (dist > 28f) continue;
                // 뒤에 있는 것은 지나간 것이다. 진행 방향 앞쪽만 센다.
                if (Vector3.Dot(v.normalized, car.forward) < 0.15f) continue;
                near.Add(i);
                if (seenNear.Contains(i)) continue;
                boot.log.Add("track", string.Format(
                    "{0} 진입  {1:0.0} m  {2}  관측 {3}회",
                    d.cls, dist, d.moving ? "이동" : "정지", d.seen),
                    LogKind.Detect);
            }
            seenNear.Clear();
            foreach (var i in near) seenNear.Add(i);
        }

        void Drive()
        {
            if (route == null || car == null) return;
            Vector3 p, f;
            if (!route.Sample(s, out p, out f)) return;
            car.position = p;
            if (f.sqrMagnitude > 1e-6f)
                car.rotation = Quaternion.LookRotation(f, Vector3.up);
        }

        void Frame()
        {
            if (cam == null || car == null) return;
            switch (view)
            {
                case View.Driver:
                    cam.transform.position = car.position + Vector3.up * 1.45f
                                           + car.forward * 0.6f;
                    cam.transform.rotation =
                        Quaternion.LookRotation(car.forward, Vector3.up);
                    cam.fieldOfView = 62f;
                    break;
                case View.Chase:
                    cam.transform.position = car.position - car.forward * 11f
                                           + Vector3.up * 5.5f;
                    cam.transform.rotation = Quaternion.LookRotation(
                        (car.position + Vector3.up * 1.2f) - cam.transform.position,
                        Vector3.up);
                    cam.fieldOfView = 55f;
                    break;
                case View.Aerial:
                {
                    // 가까이 볼 때는 자차를 따라간다. 전체를 볼 때는 장면
                    // 한가운데를 본다 - 줌을 당겼는데 화면이 안 움직이면
                    // 어디를 보고 있는지 알 수 없다.
                    float full = Mathf.Max(world.size.x, world.size.z) * 0.55f;
                    float size = Mathf.Max(6f, full * aerialZoom);
                    float t = Mathf.Clamp01((aerialZoom - 0.08f) / 0.5f);
                    Vector3 c2 = Vector3.Lerp(car.position, world.center, t);
                    cam.transform.position = new Vector3(
                        c2.x, world.max.y + 80f, c2.z);
                    cam.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
                    cam.orthographicSize = size;
                    break;
                }
            }
        }

        // ------------------------------------------------------------------
        // 화면
        // ------------------------------------------------------------------
        GUIStyle st, stDim, stHead, stNum, stMono;

        // 간격 체계
        // ---------
        // **한 줄의 높이를 글자 크기에서 정한다.** 처음에는 13 px 글자에
        // 18 px 줄을 줬는데, 한글은 라틴 글자보다 세로로 크고 받침이 기준선
        // 아래로 내려가서 "건물" 이 "거물" 로, "488 동" 이 "488 두" 로 잘렸다.
        // 라틴 문자만 보고 맞춘 간격은 한글에서 틀린다.
        //
        // 글자 크기의 1.65 배면 받침까지 들어가고 줄 사이도 답답하지 않다.
        const float kFont = 13f;
        const float kRow = 22f;      // kFont * 1.65
        const float kHeadRow = 20f;
        const float kGap = 12f;      // 묶음 사이
        const float kPadX = 18f;

        void MakeStyles()
        {
            if (st != null) return;
            st = new GUIStyle(GUI.skin.label);
            st.fontSize = (int)kFont;
            st.normal.textColor = new Color(0.88f, 0.92f, 0.97f);
            // 잘라내지 않는다. 폰트 대체로 글자가 예상보다 커져도 사라지는
            // 것보다 조금 넘치는 편이 낫다.
            st.clipping = TextClipping.Overflow;
            st.wordWrap = false;
            st.padding = new RectOffset(0, 0, 2, 2);

            stDim = new GUIStyle(st);
            stDim.normal.textColor = new Color(0.55f, 0.62f, 0.72f);

            stHead = new GUIStyle(st);
            stHead.fontSize = 12;
            stHead.normal.textColor = new Color(0.42f, 0.70f, 0.84f);

            stNum = new GUIStyle(st);
            stNum.fontSize = 26;
            stNum.normal.textColor = new Color(0.55f, 0.92f, 0.86f);

            stMono = new GUIStyle(st);
            stMono.fontSize = 12;
        }

        void Box(Rect r) { GUI.DrawTexture(r, panelTex); }

        void Swatch(float x, float y, Color c)
        {
            var g = GUI.color;
            GUI.color = c;
            GUI.DrawTexture(new Rect(x, y + 3f, 10f, 10f), barTex);
            GUI.color = g;
        }

        void Outline(Rect r, Color c)
        {
            var g = GUI.color;
            GUI.color = c;
            GUI.DrawTexture(new Rect(r.x, r.y, r.width, 1f), barTex);
            GUI.DrawTexture(new Rect(r.x, r.y + r.height - 1f, r.width, 1f), barTex);
            GUI.DrawTexture(new Rect(r.x, r.y, 1f, r.height), barTex);
            GUI.DrawTexture(new Rect(r.x + r.width - 1f, r.y, 1f, r.height), barTex);
            GUI.color = g;
        }

        float Median()
        {
            if (frameMs.Count == 0) return 0f;
            var v = new List<float>(frameMs);
            v.Sort();
            return v[v.Count / 2];
        }

        static Color Kind(LogKind k)
        {
            switch (k)
            {
                case LogKind.Ok:     return new Color(0.45f, 0.90f, 0.62f);
                case LogKind.Warn:   return new Color(0.95f, 0.80f, 0.35f);
                case LogKind.Error:  return new Color(0.95f, 0.45f, 0.40f);
                case LogKind.Detect: return new Color(0.50f, 0.80f, 0.98f);
                default:             return new Color(0.62f, 0.70f, 0.80f);
            }
        }

        void Term(Rect r)
        {
            Box(r);
            var g = GUI.color;
            GUI.color = new Color(0.20f, 0.28f, 0.38f, 0.8f);
            GUI.DrawTexture(new Rect(r.x, r.y, r.width, 1f), barTex);
            GUI.color = g;
            GUI.Label(new Rect(r.x + kPadX, r.y + 6f, 400f, kHeadRow),
                      "인식 로그", stHead);
            if (boot == null || boot.log == null) return;
            var lines = boot.log.lines;
            // 줄 높이도 한글에 맞춘다. 12 px 글자에 15 px 줄이면 받침이 잘린다.
            const float row = 18f;
            int rows = (int)((r.height - 32f) / row);
            int from = Mathf.Max(0, lines.Count - rows);
            float y = r.y + 30f;
            for (int i = from; i < lines.Count; ++i)
            {
                var L = lines[i];
                var st2 = new GUIStyle(stMono);
                st2.normal.textColor = Kind(L.kind);
                // 시각과 꼬리표를 고정된 자리에 놓는다. 비례 폭 글꼴이라
                // 문자열 안에서 칸을 맞추면 줄마다 어긋난다.
                GUI.Label(new Rect(r.x + kPadX, y, 74f, row),
                          string.Format("{0,7:0.00}", L.t), st2);
                GUI.Label(new Rect(r.x + kPadX + 80f, y, 68f, row), L.tag, st2);
                GUI.Label(new Rect(r.x + kPadX + 154f, y,
                                   r.width - kPadX - 166f, row), L.msg, st2);
                y += row;
            }
        }

        // 이름표만 그린다. **상자는 3D 로 간다** (WorldVisionLines).
        //
        // 처음에는 여덟 꼭짓점을 감싸는 화면 사각형을 그렸는데, 비스듬히 선
        // 차에서는 그 사각형이 차보다 훨씬 넓어 상자가 과대해 보였고 물체가
        // 어느 쪽을 보는지도 사라졌다.
        void Labels()
        {
            if (!showLabels || boot == null || cam == null) return;
            float sy = Screen.height;
            foreach (var d in boot.detections)
            {
                float H2 = Mathf.Max(0.4f, d.size.y);
                Vector3 top = d.center + Vector3.up * (H2 * 0.5f + 0.15f);
                Vector3 q = cam.WorldToScreenPoint(top);
                if (q.z <= 1f || q.z > 60f) continue;
                float x = q.x, y = sy - q.y;
                if (x < kPanelW || x > Screen.width - 40f) continue;
                if (showTerminal && y > sy - kTermH) continue;
                Color c = d.moving ? new Color(0.98f, 0.60f, 0.30f)
                                   : new Color(0.35f, 0.88f, 0.98f);
                var lab = new GUIStyle(stMono);
                lab.normal.textColor = c;
                GUI.Label(new Rect(x - 40f, y - 16f, 240f, 15f),
                    string.Format("{0} {1:0.0}m {2}회{3}", d.cls, q.z,
                                  d.seen, d.moving ? " 이동" : ""), lab);
            }
        }

        void OnGUI()
        {
            MakeStyles();
            if (!help)
            {
                GUI.Label(new Rect(12, 8, 400, 20), "H  화면 표시", stDim);
                return;
            }

            const float W = kPanelW;
            float H = Screen.height;

            Labels();

            Box(new Rect(0, 0, W, H));
            // 패널 오른쪽에 얇은 경계선. 배경이 검은 3D 라 면만으로는 어디
            // 까지가 패널인지 흐릿하다.
            var gEdge = GUI.color;
            GUI.color = new Color(0.20f, 0.28f, 0.38f, 0.8f);
            GUI.DrawTexture(new Rect(W - 1f, 0f, 1f, H), barTex);
            GUI.color = gEdge;

            // **넘치면 스크롤한다.**
            //
            // 창을 낮게 잡거나 클래스가 늘면 아래쪽 "조작" 이 화면 밖으로
            // 밀린다. 항목을 줄이는 것은 답이 아니다 - 조작키는 늘 필요하고,
            // 클래스 수는 데이터가 정한다. 내용 높이를 재서 넘칠 때만
            // 스크롤 막대를 둔다.
            bool over = panelH > H;
            float innerW = over ? W - 16f : W;
            if (over)
            {
                panelScroll = GUI.BeginScrollView(
                    new Rect(0f, 0f, W, H), panelScroll,
                    new Rect(0f, 0f, innerW, panelH), false, true);
            }

            float y = 18f;
            GUI.Label(new Rect(kPadX, y, W - kPadX * 2f, 26f),
                      "WORLDVISION—SLAM", st);
            y += 24f;
            GUI.Label(new Rect(kPadX, y, W - kPadX * 2f, kRow),
                      "카메라로 지은 주행 시뮬레이션", stDim);
            y += kRow;

            Head(ref y, "시퀀스");
            string seq = stats != null ? stats.sequence : "—";
            int fr = stats != null ? stats.frame : 0;
            GUI.Label(new Rect(kPadX, y, W - kPadX * 2f, kRow),
                string.Format("{0}   누적 {1} 프레임", seq, fr), st);
            y += kRow;

            Head(ref y, "주행");
            GUI.Label(new Rect(kPadX, y, 170f, 34f),
                      string.Format("{0:n0} m", s), stNum);
            GUI.Label(new Rect(kPadX + 140f, y + 12f, 170f, kRow),
                      string.Format("/ {0:n0} m", routeLen), stDim);
            y += 38f;
            GUI.Label(new Rect(kPadX, y, W - kPadX * 2f, kRow), string.Format(
                "{0:0.0} m/s      {1:0} km/h      {2}",
                speed, speed * 3.6f, playing ? "재생" : "정지"), st);
            y += kRow + 4f;

            var track = new Rect(kPadX, y, W - kPadX * 2f, 6f);
            var g0 = GUI.color;
            GUI.color = new Color(1f, 1f, 1f, 0.12f);
            GUI.DrawTexture(track, barTex);
            GUI.color = new Color(0.42f, 0.86f, 0.80f, 0.95f);
            GUI.DrawTexture(new Rect(track.x, track.y,
                track.width * Mathf.Clamp01(s / routeLen), track.height), barTex);
            GUI.color = g0;
            y += 12f;

            Head(ref y, "인식한 것");
            if (stats != null)
            {
                float a = stats.surfaceCell * stats.surfaceCell;
                Row(ref y, cRoad, "도로", stats.tRoad, a, "m2");
                Row(ref y, cSide, "인도", stats.tSidewalk, a, "m2");
                Row(ref y, cGrass, "잔디", stats.tGrass, a, "m2");
                Row(ref y, cLane, "차선 (예측)", stats.lanes, 0f, "줄");
                y += 4f;
                Row(ref y, cBuild, "건물", stats.buildings, 0f, "동");
                Row(ref y, cTree, "나무", stats.trees, 0f, "그루");
                Row(ref y, cCar, "차량", stats.vehicles, 0f, "대");
            }

            if (boot != null)
            {
                Head(ref y, "점군 색      C 로 전환");
                GUI.Label(new Rect(kPadX, y, W - kPadX * 2f, kRow),
                    boot.paint == Boot.Paint.Height ? "지면 위 높이"
                    : boot.paint == Boot.Paint.Intensity ? "관측 밝기" : "분류",
                    st);
                y += kRow;
                // 색이 무엇을 뜻하는지 눈금으로 보인다. 색만 있고 자가 없으면
                // "파란 쪽이 낮다" 는 것 말고는 읽을 수 없다.
                Ramp(new Rect(kPadX, y, W - kPadX * 2f, 9f));
                y += 13f;
                var hi = new GUIStyle(stDim);
                hi.alignment = TextAnchor.UpperRight;
                string a0, a1;
                if (boot.paint == Boot.Paint.Height) { a0 = "0 m"; a1 = "12 m"; }
                else if (boot.paint == Boot.Paint.Intensity) { a0 = "어둠"; a1 = "밝음"; }
                else { a0 = "지면"; a1 = "기둥"; }
                GUI.Label(new Rect(kPadX, y, 140f, kRow), a0, stDim);
                GUI.Label(new Rect(W - kPadX - 140f, y, 140f, kRow), a1, hi);
                y += kRow;
            }

            // 클래스 표시/숨김. 레퍼런스의 눈 아이콘과 같은 자리다.
            if (boot != null && boot.byClass[0] != null)
            {
                Head(ref y, "점군 표시      F1~F7");
                var right2 = new GUIStyle(st);
                right2.alignment = TextAnchor.UpperRight;
                var rightDim = new GUIStyle(stDim);
                rightDim.alignment = TextAnchor.UpperRight;
                for (int c = 0; c < Boot.kClassN; ++c)
                {
                    if (boot.classCount[c] == 0) continue;
                    bool on = boot.classOn[c];
                    Swatch(kPadX, y, on
                        ? (Color)Boot.Shade(Boot.Paint.Class, 0f, 230, (byte)c)
                        : new Color(0.18f, 0.20f, 0.24f));
                    GUI.Label(new Rect(kPadX + 20f, y, 160f, kRow),
                        "F" + (c + 1) + "   " + Boot.kClassName[c],
                        on ? st : stDim);
                    GUI.Label(new Rect(W - kPadX - 130f, y, 130f, kRow),
                        string.Format("{0:n0}", boot.classCount[c]),
                        on ? right2 : rightDim);
                    y += kRow;
                }
            }

            Head(ref y, "성능");
            float ms = Median();
            GUI.Label(new Rect(kPadX, y, W - kPadX * 2f, kRow), string.Format(
                "{0:0.0} ms / 프레임        {1:0} fps",
                ms, ms > 0f ? 1000f / ms : 0f), st);
            y += kRow;
            if (boot != null)
            {
                GUI.Label(new Rect(kPadX, y, W - kPadX * 2f, kRow),
                    string.Format("점군 {0:n0} 점", boot.cloudPoints), stDim);
                y += kRow;
            }

            Head(ref y, "보기");
            ViewRow(ref y, View.Driver, "1    운전석");
            ViewRow(ref y, View.Chase, "2    추적");
            ViewRow(ref y, View.Aerial, view == View.Aerial && cam != null
                ? string.Format("3    항공   폭 {0:n0} m", cam.orthographicSize * 2f)
                : "3    항공");
            ViewRow(ref y, View.Free, "4    자유 · WASD 마우스");

            Head(ref y, "조작");
            GUI.Label(new Rect(kPadX, y, W - kPadX * 2f, kRow),
                "Space   재생 · 정지", stDim); y += kRow;
            GUI.Label(new Rect(kPadX, y, W - kPadX * 2f, kRow),
                "← →   위치          [ ]   속도", stDim); y += kRow;
            GUI.Label(new Rect(kPadX, y, W - kPadX * 2f, kRow),
                "L 라벨      T 터미널      C 색", stDim); y += kRow;
            GUI.Label(new Rect(kPadX, y, W - kPadX * 2f, kRow),
                "X 투과 보기" + (xray ? "  켜짐" : "") + "      휠  항공 줌",
                stDim); y += kRow;
            GUI.Label(new Rect(kPadX, y, W - kPadX * 2f, kRow),
                "R 처음      H 표시      Q 종료", stDim); y += kRow;

            // 다음 프레임의 스크롤 여부를 정할 높이. 그리면서 잰다 - 항목이
            // 데이터에 따라 늘고 줄기 때문에 미리 계산할 수가 없다.
            panelH = y + 16f;
            if (over) GUI.EndScrollView();

            if (showTerminal)
                Term(new Rect(W, H - kTermH, Screen.width - W - kCamW, kTermH));

            var camRect = new Rect(Screen.width - kCamW, H - kTermH, kCamW, kTermH);
            Box(camRect);
            GUI.Label(new Rect(camRect.x + 12, camRect.y + 5, 300, 16),
                "카메라", stHead);
            Texture2D shot = boot != null
                ? boot.CamFrame(s / Mathf.Max(1f, routeLen)) : null;
            if (shot != null)
            {
                float ar = (float)shot.width / Mathf.Max(1, shot.height);
                float w = kCamW - 24f, hgt = w / ar;
                GUI.DrawTexture(new Rect(camRect.x + 12, camRect.y + 24, w, hgt),
                                shot, ScaleMode.ScaleToFit);
            }
            else
            {
                GUI.Label(new Rect(camRect.x + 12, camRect.y + 26, kCamW - 24, 60),
                    "프레임 이미지 없음", stDim);
            }

            if (view == View.Aerial && cam != null)
            {
                float mPerPx = (cam.orthographicSize * 2f) / Screen.height;
                float px = 50f / Mathf.Max(1e-6f, mPerPx);
                float bx = W + 24f, by = H - kTermH - 34f;
                GUI.color = new Color(1f, 1f, 1f, 0.85f);
                GUI.DrawTexture(new Rect(bx, by, px, 2f), barTex);
                GUI.DrawTexture(new Rect(bx, by - 4f, 2f, 10f), barTex);
                GUI.DrawTexture(new Rect(bx + px - 2f, by - 4f, 2f, 10f), barTex);
                GUI.color = g0;
                GUI.Label(new Rect(bx, by + 4f, 120f, 20f), "50 m", stDim);
            }
        }

        // 색 눈금.
        void Ramp(Rect r)
        {
            if (boot == null) return;
            int n = 48;
            var g = GUI.color;
            for (int i = 0; i < n; ++i)
            {
                float t = i / (float)(n - 1);
                Color32 c;
                if (boot.paint == Boot.Paint.Height)
                    c = Boot.Shade(Boot.Paint.Height, t * 12f, 200, 0);
                else if (boot.paint == Boot.Paint.Intensity)
                    c = Boot.Shade(Boot.Paint.Intensity, 0f, (byte)(t * 255f), 0);
                else
                {
                    byte[] order = { 1, 2, 4, 6, 5 };
                    c = Boot.Shade(Boot.Paint.Class, 0f, 220,
                                   order[Mathf.Min(order.Length - 1,
                                         (int)(t * order.Length))]);
                }
                GUI.color = c;
                GUI.DrawTexture(new Rect(r.x + r.width * i / n, r.y,
                                         r.width / n + 1f, r.height), barTex);
            }
            GUI.color = g;
        }

        // 값을 오른쪽에 맞춘다. 왼쪽 정렬하면 자리수가 다른 숫자가 들쭉
        // 날쭉해서 세로로 훑을 수가 없다.
        void Row(ref float y, Color c, string name, int n, float area, string unit)
        {
            Swatch(kPadX, y, c);
            GUI.Label(new Rect(kPadX + 20f, y, 150f, kRow), name, st);
            string v = area > 0f
                ? string.Format("{0:n0} {1}", n * area, unit)
                : string.Format("{0:n0} {1}", n, unit);
            var right = new GUIStyle(st);
            right.alignment = TextAnchor.UpperRight;
            GUI.Label(new Rect(kPanelW - kPadX - 130f, y, 130f, kRow), v, right);
            y += kRow;
        }

        void ViewRow(ref float y, View v, string label)
        {
            bool on = view == v;
            if (on) Swatch(kPadX, y, new Color(0.42f, 0.86f, 0.80f));
            GUI.Label(new Rect(kPadX + 20f, y, kPanelW - kPadX * 2f - 20f, kRow),
                      label, on ? st : stDim);
            y += kRow;
        }

        // 묶음 제목. 위에 여백을 두고 아래는 붙인다 - 제목이 아래 내용에
        // 붙어야 어느 묶음의 제목인지가 보인다.
        void Head(ref float y, string text)
        {
            y += kGap;
            GUI.Label(new Rect(kPadX, y, kPanelW - kPadX * 2f, kHeadRow), text, stHead);
            y += kHeadRow;
        }
    }
}
