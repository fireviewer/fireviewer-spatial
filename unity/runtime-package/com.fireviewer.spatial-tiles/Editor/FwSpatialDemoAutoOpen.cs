using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace FireViewer.SpatialTiles.Editor
{
    /// <summary>
    /// Opens the generated FireViewer demo when the editor starts on an empty
    /// Untitled scene. Existing user scenes are never replaced.
    /// </summary>
    [InitializeOnLoad]
    internal static class FwSpatialDemoAutoOpen
    {
        private const string OpenSessionKey = "FireViewer.SpatialTiles.DemoAutoOpenAttempted";
        private const string PlaySessionKey = "FireViewer.SpatialTiles.DemoAutoPlayAttempted";

        static FwSpatialDemoAutoOpen()
        {
            EditorApplication.delayCall += TryOpenAndPlayGeneratedDemo;
        }

        private static void TryOpenAndPlayGeneratedDemo()
        {
            if (Application.isBatchMode || EditorApplication.isCompiling || EditorApplication.isUpdating)
                return;

            Scene active = SceneManager.GetActiveScene();
            if (string.IsNullOrEmpty(active.path))
            {
                if (SessionState.GetBool(OpenSessionKey, false) ||
                    AssetDatabase.LoadAssetAtPath<SceneAsset>(FwSpatialDemoSceneBuilder.ScenePath) == null)
                    return;

                SessionState.SetBool(OpenSessionKey, true);
                EditorSceneManager.OpenScene(FwSpatialDemoSceneBuilder.ScenePath, OpenSceneMode.Single);
                Debug.Log($"FIREVIEWER_SPATIAL_DEMO_OPENED scene={FwSpatialDemoSceneBuilder.ScenePath}");
                EditorApplication.delayCall += TryOpenAndPlayGeneratedDemo;
                return;
            }

            if (!string.Equals(active.path, FwSpatialDemoSceneBuilder.ScenePath, System.StringComparison.Ordinal) ||
                SessionState.GetBool(PlaySessionKey, false) || EditorApplication.isPlayingOrWillChangePlaymode)
                return;

            SessionState.SetBool(PlaySessionKey, true);
            Debug.Log($"FIREVIEWER_SPATIAL_DEMO_PLAY_REQUESTED scene={FwSpatialDemoSceneBuilder.ScenePath}");
            EditorApplication.isPlaying = true;
        }
    }
}
