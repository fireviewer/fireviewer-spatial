Shader "FireViewer/Operational Terrain"
{
    Properties
    {
        _MainTex ("Orthophoto", 2D) = "white" {}
        _BaseMap ("Orthophoto (SRP)", 2D) = "white" {}
        _Tint ("Tint", Color) = (1, 1, 1, 1)
        _ClipDetailFootprints ("Clip active detail footprints", Float) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        LOD 150
        Cull Back
        ZWrite On
        ZTest LEqual

        Pass
        {
            CGPROGRAM
            #pragma target 3.0
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_fog
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_ST;
            fixed4 _Tint;
            float _ClipDetailFootprints;
            int _FwDetailBoundsCount;
            float4 _FwDetailBounds[16];

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 position : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 worldNormal : TEXCOORD1;
                float2 localXZ : TEXCOORD2;
                UNITY_FOG_COORDS(3)
            };

            v2f vert(appdata input)
            {
                v2f output;
                output.position = UnityObjectToClipPos(input.vertex);
                output.uv = TRANSFORM_TEX(input.uv, _MainTex);
                output.worldNormal = UnityObjectToWorldNormal(input.normal);
                output.localXZ = input.vertex.xz;
                UNITY_TRANSFER_FOG(output, output.position);
                return output;
            }

            fixed4 frag(v2f input) : SV_Target
            {
                if (_ClipDetailFootprints > 0.5)
                {
                    float outside = 1.0;
                    [loop]
                    for (int index = 0; index < _FwDetailBoundsCount; index++)
                    {
                        float4 bounds = _FwDetailBounds[index];
                        float inside = step(bounds.x, input.localXZ.x) * step(input.localXZ.x, bounds.z) *
                            step(bounds.y, input.localXZ.y) * step(input.localXZ.y, bounds.w);
                        outside *= 1.0 - inside;
                    }
                    clip(outside - 0.5);
                }

                fixed3 colour = tex2D(_MainTex, input.uv).rgb * _Tint.rgb;
                fixed luminance = dot(colour, fixed3(0.2126, 0.7152, 0.0722));
                colour = lerp(luminance.xxx, colour, 0.90);
                colour = (colour - 0.5) * 0.94 + 0.5;
                float3 normal = normalize(input.worldNormal);
                float relief = saturate(dot(normal, normalize(float3(0.34, 0.82, 0.46))) * 0.5 + 0.5);
                colour *= lerp(0.82, 1.03, relief);
                fixed4 result = fixed4(saturate(colour), 1.0);
                UNITY_APPLY_FOG(input.fogCoord, result);
                return result;
            }
            ENDCG
        }
    }
    Fallback "Unlit/Texture"
}
