// 아래쪽 터미널에 흐르는 줄들.
//
// 그림만 보여 주면 무엇을 근거로 그렸는지가 안 보인다. 인식이 일어난
// 순간을 글로 남기면, 화면의 상자가 어디서 왔는지 따라갈 수 있다.
//
// **제 파일에 혼자 둔다.** 원래 Boot 와 함께 WorldVisionBoot.cs 에 있었다.
// 이쪽은 씬에 직렬화되지 않고 (Boot 가 실행할 때 AddComponent 한다) 지금까지
// 탈이 없었지만, 한 파일에 MonoBehaviour 가 둘이면 어느 쪽이 그 파일의
// MonoScript 를 가져갈지가 우리 손을 떠난다. Boot 는 씬에 직렬화되므로 그
// 추첨에 걸리면 안 된다 - 까닭은 WorldVisionRoute.cs 의 주석 참조.
using System.Collections.Generic;
using UnityEngine;

namespace WorldVision
{
    public enum LogKind { Info, Ok, Warn, Error, Detect }

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
