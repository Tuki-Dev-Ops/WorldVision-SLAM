// 무엇이 몇 개 인식되었는가.
//
// 제 파일에 혼자 둔다 - 까닭은 WorldVisionRoute.cs 의 주석 참조.
using UnityEngine;

namespace WorldVision
{
    public class Stats : MonoBehaviour
    {
        public string sequence = "";
        public int frame;
        public int buildings, trees, poles, vehicles, lanes;
        public int tRoad, tSidewalk, tGrass, tOther;
        public float surfaceCell = 0.5f;
        public string roadMap = "";
    }
}
