// 점군 셰이더. 조명 없이 정점 색을 그대로 낸다.
//
// LiDAR 지도가 그렇게 보이는 이유는 점마다 색이 실려 있고 그 색이 조명이
// 아니라 **측정값** 이기 때문이다. Lit 셰이더를 쓰면 그 색에 조명이 곱해져
// 측정이 아니라 렌더링 결과가 되고, 어두운 쪽 절반이 통째로 검게 죽는다.
//
// 안개도 끈다. 거리에 따라 색이 바래면 높이 색이 거리 색과 섞인다.
Shader "WorldVision/Point"
{
    Properties
    {
        _PointSize ("점 크기 (px)", Range(1, 8)) = 2
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
            #pragma fragment frag
            #pragma target 2.0
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                fixed4 color  : COLOR;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 pos   : SV_POSITION;
                fixed4 color : COLOR;
                // 점 크기는 플랫폼마다 지원이 다르다. D3D 는 PSIZE 를 무시
                // 하므로, 크게 보이게 하려면 지오메트리 셰이더로 사각형을
                // 만들어야 한다 - 83 만 점에 그것을 돌리면 느려진다.
                // 1 px 점이 LiDAR 지도의 원래 모습이기도 하다.
                UNITY_VERTEX_OUTPUT_STEREO
            };

            v2f vert (appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.pos = UnityObjectToClipPos(v.vertex);
                o.color = v.color;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                return i.color;
            }
            ENDCG
        }
    }
    Fallback Off
}
