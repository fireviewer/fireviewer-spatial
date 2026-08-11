using UnityEngine;

namespace FireViewer.SpatialTiles
{
    [DisallowMultipleComponent]
    [AddComponentMenu("FireViewer/Spatial Scene Bootstrap")]
    public sealed class FwSpatialSceneBootstrap : MonoBehaviour
    {
        [SerializeField] private string catalogUrl = string.Empty;
        [SerializeField] private Camera viewerCamera;
        [SerializeField] private bool useCameraAsFocus;
        [SerializeField] private string initialFocusZone = "montmaur";
        [SerializeField] private bool detailNavigationAuthorized;
        [SerializeField] private bool adminEditingAuthorized;
        [SerializeField, Min(0.0001f)] private float unityUnitsPerMetre = 1f;
        [SerializeField, Range(1, FwSpatialLodPlanner.AbsoluteMaximumResidentTiles)] private int residentBudget = 16;
        [SerializeField] private FwSpatialTileStreamingController controller;
        [SerializeField] private FwSpatialCameraController cameraController;

        public FwSpatialTileStreamingController Controller => controller;

        public void Configure(
            string remoteCatalogUrl,
            Camera camera = null,
            string focusZone = "montmaur",
            bool authorizeDetailNavigation = false,
            bool authorizeAdminEditing = false)
        {
            catalogUrl = remoteCatalogUrl ?? string.Empty;
            viewerCamera = camera;
            initialFocusZone = focusZone ?? string.Empty;
            detailNavigationAuthorized = authorizeDetailNavigation;
            adminEditingAuthorized = authorizeAdminEditing;
            BuildOrConfigure();
        }

        public static FwSpatialSceneBootstrap Create(
            string remoteCatalogUrl,
            Camera camera = null,
            string focusZone = "montmaur",
            bool authorizeDetailNavigation = false,
            bool authorizeAdminEditing = false)
        {
            var root = new GameObject("FireViewer Spatial Runtime");
            FwSpatialSceneBootstrap bootstrap = root.AddComponent<FwSpatialSceneBootstrap>();
            bootstrap.Configure(remoteCatalogUrl, camera, focusZone, authorizeDetailNavigation, authorizeAdminEditing);
            return bootstrap;
        }

        /// <summary>
        /// Called by the authenticated host when the session changes.  The
        /// runtime intentionally has no independent login or bypass.
        /// </summary>
        public void SetDetailNavigationAuthorized(bool authorized)
        {
            detailNavigationAuthorized = authorized;
            cameraController?.SetDetailNavigationAuthorized(authorized);
            controller?.SetDetailStreamingAuthorized(authorized);
        }

        /// <summary>
        /// Called only after the host has resolved the authenticated user's
        /// administrator role.  It never grants detail access by itself.
        /// </summary>
        public void SetAdminEditingAuthorized(bool authorized)
        {
            adminEditingAuthorized = authorized;
            cameraController?.SetAdminEditingAuthorized(authorized);
        }

        private void Awake() => BuildOrConfigure();

        private void BuildOrConfigure()
        {
            // Migrate scenes created with the historical 100-units-per-metre
            // convention.  The streamed contract and camera now operate in
            // metres, including already-serialized validation scenes.
            unityUnitsPerMetre = 1f;
            if (viewerCamera != null)
            {
                viewerCamera.nearClipPlane = 2f;
                viewerCamera.farClipPlane = 50_000f;
            }
            Transform frame = Child("Local Frame ENU");
            Transform content = Child("Streamed Content");
            Transform target = useCameraAsFocus && viewerCamera != null ? viewerCamera.transform : Child("Focus Target");
            if (controller == null) controller = gameObject.GetComponent<FwSpatialTileStreamingController>() ?? gameObject.AddComponent<FwSpatialTileStreamingController>();
            controller.Configure(
                catalogUrl,
                target,
                frame,
                content,
                unityUnitsPerMetre,
                residentBudget,
                detailNavigationAuthorized);
            controller.SetViewerCamera(viewerCamera);
            if (viewerCamera != null)
            {
                cameraController = viewerCamera.GetComponent<FwSpatialCameraController>() ?? viewerCamera.gameObject.AddComponent<FwSpatialCameraController>();
                cameraController.Configure(
                    viewerCamera,
                    target,
                    controller,
                    unityUnitsPerMetre,
                    detailNavigationAuthorized,
                    adminEditingAuthorized);
            }
            if (!string.IsNullOrWhiteSpace(initialFocusZone)) controller.FocusZone(initialFocusZone);
        }

        private Transform Child(string name)
        {
            Transform existing = transform.Find(name);
            if (existing != null) return existing;
            var child = new GameObject(name);
            child.transform.SetParent(transform, false);
            return child.transform;
        }
    }
}
