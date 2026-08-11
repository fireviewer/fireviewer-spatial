Shader "FireViewer/Operational Vegetation"
{
    Properties
    {
        _Color ("Vegetation colour", Color) = (0.14, 0.27, 0.14, 1)
        _Foliage ("Foliage material", Range(0, 1)) = 1
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry+10" }
        LOD 140
        Cull Back
        ZWrite On
        ZTest LEqual

        Pass
        {
            CGPROGRAM
            #pragma target 3.0
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #pragma multi_compile_fog
            #include "UnityCG.cginc"

            fixed4 _Color;
            float _Foliage;

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 position : SV_POSITION;
                float3 worldNormal : TEXCOORD0;
                float3 worldPosition : TEXCOORD1;
                float localHeight : TEXCOORD2;
                float instanceVariation : TEXCOORD3;
                UNITY_FOG_COORDS(4)
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            float InstanceVariation(float3 origin)
            {
                return frac(sin(dot(origin.xz, float2(0.071, 0.113))) * 43758.5453);
            }

            v2f vert(appdata input)
            {
                UNITY_SETUP_INSTANCE_ID(input);
                v2f output;
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                output.position = UnityObjectToClipPos(input.vertex);
                output.worldNormal = UnityObjectToWorldNormal(input.normal);
                output.worldPosition = mul(unity_ObjectToWorld, input.vertex).xyz;
                output.localHeight = input.vertex.y;
                output.instanceVariation = InstanceVariation(mul(unity_ObjectToWorld, float4(0, 0, 0, 1)).xyz);
                UNITY_TRANSFER_FOG(output, output.position);
                return output;
            }

            fixed4 frag(v2f input) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(input);
                float3 normal = normalize(input.worldNormal);
                float3 sunDirection = normalize(float3(0.30, 0.86, 0.40));
                float direct = saturate(dot(normal, sunDirection) * 0.5 + 0.5);
                float hemisphere = saturate(normal.y * 0.5 + 0.5);
                float lighting = 0.48 + direct * 0.34 + hemisphere * 0.18;

                float crownGradient = lerp(0.82, 1.10, smoothstep(0.18, 1.0, input.localHeight));
                float materialGradient = lerp(0.88 + 0.10 * saturate(input.localHeight), crownGradient, _Foliage);
                float variation = lerp(0.93, 1.07, input.instanceVariation);
                float3 viewDirection = normalize(_WorldSpaceCameraPos.xyz - input.worldPosition);
                float silhouette = smoothstep(0.76, 0.98, 1.0 - abs(dot(normal, viewDirection)));

                fixed3 colour = saturate(_Color.rgb * lighting * materialGradient * variation);
                colour *= lerp(1.0, 0.84, silhouette * _Foliage);
                fixed4 result = fixed4(colour, _Color.a);
                UNITY_APPLY_FOG(input.fogCoord, result);
                return result;
            }
            ENDCG
        }
    }
    Fallback "FireViewer/Operational Feature"
}
