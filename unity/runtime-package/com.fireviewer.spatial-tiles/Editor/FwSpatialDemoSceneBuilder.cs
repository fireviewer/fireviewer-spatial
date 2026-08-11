using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

namespace FireViewer.SpatialTiles.Editor
{
    public static class FwSpatialDemoSceneBuilder
    {
        public const string ScenePath = "Assets/FireViewerSpatial/FireViewerSpatialDemo.unity";
        private const string LastCatalogUrlKey = "FireViewer.SpatialTiles.LastCatalogUrl";

        [MenuItem("FireViewer/Create or Replace Spatial Demo Scene")]
        public static void CreateFromMenu() => CreateScene(EditorPrefs.GetString(LastCatalogUrlKey, "http://127.0.0.1:8000/catalog.json"));

        public static void CreateFromCommandLine()
        {
            string catalogUrl = CommandLineValue("-fireviewerCatalogUrl") ?? EditorPrefs.GetString(LastCatalogUrlKey, "http://127.0.0.1:8000/catalog.json");
            CreateScene(catalogUrl);
        }

        public static void CreateScene(string catalogUrl)
        {
            if (!Uri.TryCreate(catalogUrl, UriKind.Absolute, out _)) throw new ArgumentException("FireViewer catalog URL must be absolute.");
            EditorPrefs.SetString(LastCatalogUrlKey, catalogUrl);
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var runtimeRoot = new GameObject("FireViewer Spatial Runtime");
            FwSpatialSceneBootstrap bootstrap = runtimeRoot.AddComponent<FwSpatialSceneBootstrap>();

            var cameraObject = new GameObject("FireViewer Camera");
            cameraObject.tag = "MainCamera";
            Camera camera = cameraObject.AddComponent<Camera>();
            camera.nearClipPlane = 2f;
            camera.farClipPlane = 50_000f;
            camera.fieldOfView = 55f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.12f, 0.15f, 0.17f);
            camera.allowHDR = true;
            // The editor scene is an explicit local validation harness.  A
            // production host leaves this false until its authenticated
            // session authorizes detailed navigation.
            bootstrap.Configure(catalogUrl, camera, string.Empty, true, true);
            Transform target = runtimeRoot.transform.Find("Focus Target");
            camera.transform.SetParent(target, false);
            camera.transform.localPosition = new Vector3(0f, 8_485f, -8_485f);
            camera.transform.localRotation = Quaternion.Euler(35f, 0f, 0f);

            var lightObject = new GameObject("Sun");
            Light light = lightObject.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 0.85f;
            light.color = new Color(1f, 0.97f, 0.92f);
            light.shadows = LightShadows.Soft;
            light.shadowStrength = 0.45f;
            light.transform.rotation = Quaternion.Euler(48f, -28f, 0f);

            RenderSettings.ambientMode = AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.42f, 0.46f, 0.48f);

            Directory.CreateDirectory(Path.GetDirectoryName(ScenePath) ?? "Assets");
            EditorSceneManager.SaveScene(scene, ScenePath);
            Selection.activeGameObject = runtimeRoot;
            Debug.Log($"FIREVIEWER_SPATIAL_DEMO_CREATED scene={ScenePath} catalog={catalogUrl} initialFocus=global viewDistanceM=12000 detailNavigationAuthorized=editor_validation_only adminFpsAuthorized=editor_validation_only");
        }

        private static string CommandLineValue(string name)
        {
            string[] arguments = Environment.GetCommandLineArgs();
            for (int index = 0; index + 1 < arguments.Length; index++)
                if (string.Equals(arguments[index], name, StringComparison.Ordinal)) return arguments[index + 1];
            return null;
        }
    }

    [CustomEditor(typeof(FwSpatialSceneBootstrap))]
    internal sealed class FwSpatialSceneBootstrapInspector : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            FwSpatialSceneBootstrap bootstrap = (FwSpatialSceneBootstrap)target;
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Zones de focus", EditorStyles.boldLabel);
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Montmaur")) bootstrap.Controller?.FocusMontmaur();
                if (GUILayout.Button("Barsac")) bootstrap.Controller?.FocusBarsac();
                if (GUILayout.Button("Ausson")) bootstrap.Controller?.FocusAusson();
            }
        }
    }
}
