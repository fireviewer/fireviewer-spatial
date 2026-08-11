Shader "FireViewer/Operational Building"
{
    Properties
    {
        _RoofColor ("Roof colour", Color) = (0.69, 0.62, 0.53, 1)
        _WallColor ("Wall colour", Color) = (0.43, 0.39, 0.35, 1)
        _EdgeColor ("Silhouette colour", Color) = (0.13, 0.15, 0.16, 1)
        _WindowColor ("Near facade detail", Color) = (0.10, 0.14, 0.16, 1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry+20" }
        LOD 160

        Pass
        {
            Name "BUILDING_FILL"
            Cull Back
            ZWrite On
            ZTest LEqual

            CGPROGRAM
            #pragma target 3.0
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #pragma multi_compile_fog
            #include "UnityCG.cginc"

            fixed4 _RoofColor;
            fixed4 _WallColor;
            fixed4 _EdgeColor;
            fixed4 _WindowColor;

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
                UNITY_FOG_COORDS(2)
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            v2f vert(appdata input)
            {
                UNITY_SETUP_INSTANCE_ID(input);
                v2f output;
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                output.position = UnityObjectToClipPos(input.vertex);
                output.worldNormal = UnityObjectToWorldNormal(input.normal);
                output.worldPosition = mul(unity_ObjectToWorld, input.vertex).xyz;
                UNITY_TRANSFER_FOG(output, output.position);
                return output;
            }

            fixed4 frag(v2f input) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(input);
                float3 normal = normalize(input.worldNormal);
                float roof = smoothstep(0.62, 0.88, abs(normal.y));
                float light = saturate(dot(normal, normalize(float3(0.30, 0.86, 0.40))) * 0.5 + 0.5);
                fixed4 baseColour = lerp(_WallColor, _RoofColor, roof);
                float3 viewDirection = normalize(_WorldSpaceCameraPos.xyz - input.worldPosition);
                float silhouette = smoothstep(0.72, 0.96, 1.0 - abs(dot(normal, viewDirection)));
                fixed3 litColour = saturate(baseColour.rgb * lerp(0.68, 1.04, light));
                float facadeAxis = abs(normal.x) > abs(normal.z) ? input.worldPosition.z : input.worldPosition.x;
                float windowColumn = frac(facadeAxis / 2.4);
                float windowRow = frac(input.worldPosition.y / 2.8);
                float columnMask = smoothstep(0.14, 0.20, windowColumn) * (1.0 - smoothstep(0.72, 0.78, windowColumn));
                float rowMask = smoothstep(0.20, 0.27, windowRow) * (1.0 - smoothstep(0.66, 0.73, windowRow));
                float nearFade = 1.0 - smoothstep(280.0, 650.0, distance(_WorldSpaceCameraPos.xyz, input.worldPosition));
                float windowMask = columnMask * rowMask * (1.0 - roof) * nearFade;
                litColour = lerp(litColour, _WindowColor.rgb, windowMask * 0.88);
                fixed4 result = fixed4(lerp(litColour, _EdgeColor.rgb, silhouette * 0.72), baseColour.a);
                UNITY_APPLY_FOG(input.fogCoord, result);
                return result;
            }
            ENDCG
        }

    }
    Fallback "FireViewer/Operational Feature"
}
