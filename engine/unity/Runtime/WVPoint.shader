// 점군 셰이더. 조명 없이 정점 색을 그대로 낸다.
//
// LiDAR 지도가 그렇게 보이는 이유는 점마다 색이 실려 있고 그 색이 조명이
// 아니라 **측정값** 이기 때문이다. Lit 셰이더를 쓰면 그 색에 조명이 곱해져
// 측정이 아니라 렌더링 결과가 되고, 어두운 쪽 절반이 통째로 죽는다.
//
// 점 크기
// -------
// 1 px 점으로 찍으면 가까운 곳이 성기게 보인다. 복셀이 0.3 m 라 근거리에서
// 점 사이가 화면에서 수십 px 씩 벌어지기 때문이다. 지오메트리 셰이더로
// 화면 공간 사각형을 만들어 **거리에 따라 크기를 준다** - 가까울수록 크게,
// 멀수록 1 px 로. 그래야 가까운 면이 면으로 보이고 먼 곳은 뭉치지 않는다.
//
// 크기를 거리에 반비례시키지는 않는다. 그러면 바로 앞 점이 화면을 덮는다.
// 상한을 두고 그 사이를 부드럽게 잇는다.
Shader "WorldVision/Point"
{
    Properties
    {
        _Size ("점 크기 (m)", Range(0.02, 0.6)) = 0.11
        _MinPx ("최소 크기 (px)", Range(1, 4)) = 1.4
        _MaxPx ("최대 크기 (px)", Range(2, 24)) = 6
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" "Queue" = "Geometry" }
        Cull Off
        ZWrite On
        Lighting Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma geometry geom
            #pragma fragment frag
            #pragma target 4.0
            #include "UnityCG.cginc"

            float _Size, _MinPx, _MaxPx;

            struct appdata
            {
                float4 vertex : POSITION;
                fixed4 color  : COLOR;
            };

            struct v2g
            {
                float4 world : TEXCOORD0;
                fixed4 color : COLOR;
            };

            struct g2f
            {
                float4 pos   : SV_POSITION;
                fixed4 color : COLOR;
            };

            v2g vert (appdata v)
            {
                v2g o;
                o.world = mul(unity_ObjectToWorld, v.vertex);
                o.color = v.color;
                return o;
            }

            [maxvertexcount(4)]
            void geom(point v2g i[1], inout TriangleStream<g2f> tri)
            {
                float4 wp = i[0].world;
                float4 cp = mul(UNITY_MATRIX_VP, wp);
                if (cp.w <= 0) return;

                // 월드에서 _Size 미터인 사각형이 화면에서 몇 px 인가.
                float px = _Size * _ScreenParams.y * abs(UNITY_MATRIX_P[1][1])
                         / max(1e-4, cp.w);
                px = clamp(px, _MinPx, _MaxPx);

                float2 half = px / _ScreenParams.xy * cp.w;

                g2f o;
                o.color = i[0].color;
                float2 d[4] = {
                    float2(-half.x, -half.y), float2(-half.x,  half.y),
                    float2( half.x, -half.y), float2( half.x,  half.y)
                };
                [unroll]
                for (int k = 0; k < 4; ++k)
                {
                    o.pos = cp;
                    o.pos.xy += d[k];
                    tri.Append(o);
                }
            }

            fixed4 frag (g2f i) : SV_Target { return i.color; }
            ENDCG
        }
    }

    // 지오메트리 셰이더가 없는 환경을 위한 대비책. 1 px 점으로 떨어진다.
    SubShader
    {
        Tags { "RenderType" = "Opaque" }
        Cull Off
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 2.0
            #include "UnityCG.cginc"
            struct appdata { float4 vertex : POSITION; fixed4 color : COLOR; };
            struct v2f { float4 pos : SV_POSITION; fixed4 color : COLOR; };
            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.color = v.color;
                return o;
            }
            fixed4 frag (v2f i) : SV_Target { return i.color; }
            ENDCG
        }
    }
    Fallback Off
}
