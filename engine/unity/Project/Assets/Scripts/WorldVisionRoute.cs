// 자차가 지나간 자리.
//
// **이 파일에 MonoBehaviour 는 이것 하나뿐이다.** 원래 Route 는 Stats,
// Director 와 함께 WorldVisionSim.cs 에 있었는데, .cs 파일 하나에 붙는
// MonoScript 는 하나뿐이라 (fileID 11500000) 나머지 클래스는 씬에서 가리킬
// 에셋이 없었다. AddComponent 는 에디터 안에서 멀쩡히 되므로 아무 일도 없어
// 보이지만, 씬을 직렬화하면 guid 없는 끊긴 참조로 저장된다:
//
//     m_Script: {fileID: 574151018}        <- 끊긴 것
//     m_Script: {fileID: 11500000, guid: ..., type: 3}   <- 성한 것
//
// 그렇게 나온 빌드가 실행할 때 "level0 is corrupted / Position out of
// bounds" 로 죽었다. 클래스마다 제 파일을 주면 그 일이 없다.
using UnityEngine;

namespace WorldVision
{
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
}
