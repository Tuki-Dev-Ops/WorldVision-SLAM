// 선으로 그리는 것들 - 상자 · 차선 · 경로 · 거리 링.
//
// 처음에는 물체 라벨을 화면 정렬 사각형으로 그렸다. 그러면 상자가 실제보다
// 크게 보인다 - 비스듬히 선 4 m 짜리 차의 여덟 꼭짓점을 감싸는 화면 사각형은
// 차보다 훨씬 넓다. 물체가 어느 쪽을 보고 있는지도 사라진다.
//
// 3D 로 그리면 둘 다 해결된다. 여기서 GL 로 직접 선을 쏘는 이유는 상자 하나에
// GameObject 를 두면 수백 개가 되고, LineRenderer 는 굵기 때문에 가까운
// 선이 판때기가 되기 때문이다.
using System.Collections.Generic;
using UnityEngine;

namespace WorldVision
{
    public class Lines : MonoBehaviour
    {
        public Boot boot;
        public Transform car;
        public bool showBoxes = true;
        public bool showRings = true;
        public bool showRoute = true;

        Material mat;

        // 셰이더 없이 선을 그리려면 재질이 하나 필요하다. 엔진에 들어 있는
        // 것을 쓰면 빌드에 따로 챙길 것이 없다.
        void MakeMat()
        {
            if (mat != null) return;
            var sh = Shader.Find("Hidden/Internal-Colored");
            mat = new Material(sh);
            mat.hideFlags = HideFlags.HideAndDontSave;
            mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            mat.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
            // 깊이 검사를 켠다. 끄면 건물 뒤의 상자가 앞에 떠 보인다.
            mat.SetInt("_ZWrite", 0);
        }

        void OnRenderObject()
        {
            if (boot == null) return;
            MakeMat();
            mat.SetPass(0);
            GL.PushMatrix();
            GL.MultMatrix(Matrix4x4.identity);
            GL.Begin(GL.LINES);

            if (showRoute) Route();
            if (showBoxes) Boxes();
            Lanes();
            if (showRings && car != null) Rings(car.position);

            GL.End();
            GL.PopMatrix();
        }

        static void Seg(Vector3 a, Vector3 b, Color c)
        {
            GL.Color(c);
            GL.Vertex(a);
            GL.Vertex(b);
        }

        // 지나온 길과 앞으로 갈 길. 색을 나눠 어디까지 왔는지 보이게 한다.
        void Route()
        {
            var r = boot.route;
            if (r == null || r.points == null || r.points.Length < 2) return;
            var lift = Vector3.up * 0.08f;
            for (int i = 0; i + 1 < r.points.Length; ++i)
            {
                Seg(r.points[i] + lift, r.points[i + 1] + lift,
                    new Color(0.20f, 0.85f, 0.45f, 0.55f));
            }
        }

        // 예측한 차선. 노면보다 살짝 띄운다.
        void Lanes()
        {
            if (boot.lanes == null) return;
            foreach (var L in boot.lanes)
            {
                Color c = L.center ? new Color(0.95f, 0.86f, 0.35f, 0.85f)
                                   : new Color(0.85f, 0.88f, 0.92f, 0.75f);
                var lift = Vector3.up * 0.05f;
                for (int i = 0; i + 1 < L.pts.Count; ++i)
                {
                    // 중앙선은 파선이다. 실선은 추월 금지라는 뜻인데 그것은
                    // 관측한 것이 아니다.
                    if (L.center && (i % 2) == 1) continue;
                    Seg(L.pts[i] + lift, L.pts[i + 1] + lift, c);
                }
            }
        }

        // 인식한 물체의 3D 상자. 열두 모서리.
        void Boxes()
        {
            foreach (var d in boot.detections)
            {
                Vector3 f = d.forward.sqrMagnitude > 1e-6f
                    ? d.forward.normalized : Vector3.forward;
                Vector3 rgt = Vector3.Cross(Vector3.up, f).normalized;
                Vector3 up = Vector3.up;
                float L = Mathf.Max(0.6f, Mathf.Max(d.size.x,
                          Mathf.Max(d.size.y, d.size.z)));
                float W = Mathf.Max(0.4f, Mathf.Min(d.size.x, d.size.z));
                float H = Mathf.Max(0.4f, d.size.y);

                var p = new Vector3[8];
                for (int k = 0; k < 8; ++k)
                {
                    p[k] = d.center
                         + f * ((k & 1) == 0 ? L * 0.5f : -L * 0.5f)
                         + rgt * ((k & 2) == 0 ? W * 0.5f : -W * 0.5f)
                         + up * ((k & 4) == 0 ? H * 0.5f : -H * 0.5f);
                }
                Color c = d.moving ? new Color(0.98f, 0.55f, 0.25f, 0.95f)
                                   : new Color(0.25f, 0.85f, 0.98f, 0.85f);
                // 아래 네 개, 위 네 개, 기둥 네 개.
                int[,] e = {
                    {0,1},{1,3},{3,2},{2,0},
                    {4,5},{5,7},{7,6},{6,4},
                    {0,4},{1,5},{2,6},{3,7},
                };
                for (int i = 0; i < 12; ++i) Seg(p[e[i, 0]], p[e[i, 1]], c);

                // 앞쪽에 짧은 화살. 어느 쪽을 보고 있는지가 상자만으로는
                // 안 보인다 - 주차된 차와 마주 오는 차를 가르는 정보다.
                Vector3 nose = d.center + f * (L * 0.5f);
                Seg(nose, nose + f * 0.9f, c);
            }
        }

        // 자차 주변 거리 링.
        //
        // **이것은 관측이 아니라 자다.** 라이다 뷰어의 그 동심원은 센서의
        // 스캔선이지만 여기 지도는 스테레오에서 나왔으므로 그런 선은 없다.
        // 거리를 눈으로 재려고 그리는 것이고, 그래서 10 m 마다 굵기를 같게
        // 두고 화면에도 "거리 링" 이라고 적는다.
        void Rings(Vector3 c0)
        {
            const int kSeg = 72;
            for (int r = 10; r <= 40; r += 10)
            {
                float a = (r == 20 || r == 40) ? 0.30f : 0.18f;
                var col = new Color(0.35f, 0.75f, 0.95f, a);
                Vector3 prev = Vector3.zero;
                for (int i = 0; i <= kSeg; ++i)
                {
                    float th = i * Mathf.PI * 2f / kSeg;
                    var p = c0 + new Vector3(Mathf.Cos(th) * r, 0.06f,
                                             Mathf.Sin(th) * r);
                    if (i > 0) Seg(prev, p, col);
                    prev = p;
                }
            }
        }
    }
}
