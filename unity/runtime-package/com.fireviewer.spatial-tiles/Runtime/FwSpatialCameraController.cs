using UnityEngine;

namespace FireViewer.SpatialTiles
{
    /// <summary>
    /// Runtime navigation for the validation scene.  The camera orbits the
    /// Lambert-93 focus while the streaming controller derives its LOD from
    /// the real camera-to-focus distance.
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("FireViewer/Spatial Camera Controller")]
    public sealed class FwSpatialCameraController : MonoBehaviour
    {
        private const float NearDistanceMetres = 300f;
        // Keep the operational MID preset close enough to read roads, water and
        // vegetation while remaining above the 750 m NEAR classification.
        private const float MidDistanceMetres = 900f;
        private const float FarDistanceMetres = 12000f;
        private const float MinimumPitchDegrees = 8f;
        private const float MaximumPitchDegrees = 82f;
        private const float MinimumTerrainClearanceMetres = 5f;
        private const float MinimumFpsPitchDegrees = -85f;
        private const float MaximumFpsPitchDegrees = 85f;

        [SerializeField] private Camera viewerCamera;
        [SerializeField] private Transform focus;
        [SerializeField] private FwSpatialTileStreamingController streaming;
        [SerializeField, Min(0.0001f)] private float unityUnitsPerMetre = 1f;
        [SerializeField, Range(MinimumPitchDegrees, MaximumPitchDegrees)] private float pitchDegrees = 38f;
        [SerializeField] private float yawDegrees = -18f;
        [SerializeField, Min(25f)] private float distanceMetres = FarDistanceMetres;
        [SerializeField, Min(0.01f)] private float orbitSensitivity = 0.18f;
        [SerializeField, Min(0.01f)] private float zoomSensitivity = 0.18f;
        [SerializeField] private bool detailNavigationAuthorized;
        [SerializeField] private bool adminEditingAuthorized;
        [SerializeField, Min(1f)] private float adminFpsSpeedMetresPerSecond = 45f;
        [SerializeField, Min(0.01f)] private float adminFpsLookSensitivity = 0.16f;
        [SerializeField] private bool showValidationHud;

        private bool configured;
        private bool adminFpsMode;
        private float adminFpsPitchDegrees;
        private float adminFpsYawDegrees;

        public float DistanceMetres => distanceMetres;
        public bool DetailNavigationAuthorized => detailNavigationAuthorized;
        public bool AdminEditingAuthorized => adminEditingAuthorized;
        public bool IsAdminFpsMode => adminFpsMode;

        public void Configure(
            Camera camera,
            Transform focusTransform,
            FwSpatialTileStreamingController streamingController,
            float unitsPerMetre,
            bool authorized = false,
            bool adminAuthorized = false)
        {
            viewerCamera = camera;
            focus = focusTransform;
            streaming = streamingController;
            unityUnitsPerMetre = Mathf.Max(0.0001f, unitsPerMetre);
            detailNavigationAuthorized = authorized;
            adminEditingAuthorized = adminAuthorized;
            adminFpsMode = false;
            showValidationHud = false;
            distanceMetres = FarDistanceMetres;
            configured = viewerCamera != null && focus != null;
            if (configured)
            {
                viewerCamera.transform.SetParent(focus, false);
                ApplyPose();
            }
        }

        public void SetDetailNavigationAuthorized(bool authorized)
        {
            detailNavigationAuthorized = authorized;
            if (!authorized)
            {
                if (adminFpsMode) ExitAdminFpsMode();
                if (distanceMetres < FarDistanceMetres) SetFarView();
            }
        }

        public void SetAdminEditingAuthorized(bool authorized)
        {
            adminEditingAuthorized = authorized;
            if (!authorized && adminFpsMode) ExitAdminFpsMode();
        }

        public void SetValidationHudVisible(bool visible) => showValidationHud = visible;

        public void ToggleAdminFpsMode()
        {
            if (!RequireAdminEditing()) return;
            if (adminFpsMode) ExitAdminFpsMode();
            else EnterAdminFpsMode();
        }

        public void SetNearView()
        {
            if (!RequireDetailNavigation("near")) return;
            if (streaming?.NearLodDisabled == true)
            {
                SetViewDistance(MidDistanceMetres, "mid");
                return;
            }
            SetViewDistance(NearDistanceMetres, "near");
        }

        public void SetMidView()
        {
            if (RequireDetailNavigation("mid")) SetViewDistance(MidDistanceMetres, "mid");
        }

        public void SetFarView() => SetViewDistance(FarDistanceMetres, "far");

        private void Awake()
        {
            if (viewerCamera == null) viewerCamera = GetComponent<Camera>();
        }

        private void Update()
        {
            if (!configured || viewerCamera == null || focus == null) return;

            if (Input.GetKeyDown(KeyCode.Alpha4)) ToggleAdminFpsMode();
            if (adminFpsMode)
            {
                UpdateAdminFpsMode();
                return;
            }

            if (Input.GetMouseButton(1))
            {
                yawDegrees += Input.GetAxis("Mouse X") * orbitSensitivity * 100f;
                pitchDegrees = Mathf.Clamp(
                    pitchDegrees - Input.GetAxis("Mouse Y") * orbitSensitivity * 100f,
                    MinimumPitchDegrees,
                    MaximumPitchDegrees);
            }

            float wheel = Input.GetAxis("Mouse ScrollWheel");
            if (detailNavigationAuthorized && Mathf.Abs(wheel) > 0.0001f)
                distanceMetres = Mathf.Clamp(
                    distanceMetres * Mathf.Exp(-wheel * zoomSensitivity * 10f),
                    25f,
                    25_000f);

            if (Input.GetKeyDown(KeyCode.Alpha1)) SetFarView();
            if (Input.GetKeyDown(KeyCode.Alpha2)) SetMidView();
            if (Input.GetKeyDown(KeyCode.Alpha3)) SetNearView();
            if (Input.GetKeyDown(KeyCode.M)) streaming?.FocusMontmaur();
            if (Input.GetKeyDown(KeyCode.B)) streaming?.FocusBarsac();
            if (Input.GetKeyDown(KeyCode.A)) streaming?.FocusAusson();

            PanFocus();
            ApplyPose();
        }

        private void PanFocus()
        {
            float horizontal = Input.GetAxisRaw("Horizontal");
            float vertical = Input.GetAxisRaw("Vertical");
            if (Mathf.Abs(horizontal) < 0.001f && Mathf.Abs(vertical) < 0.001f)
                return;

            Vector3 forward = Vector3.ProjectOnPlane(viewerCamera.transform.forward, Vector3.up).normalized;
            if (forward.sqrMagnitude < 0.01f) forward = Vector3.forward;
            Vector3 right = Vector3.Cross(Vector3.up, forward).normalized;
            float speedMetresPerSecond = Mathf.Max(20f, distanceMetres * 0.45f);
            Vector3 movementMetres = (right * horizontal + forward * vertical) * speedMetresPerSecond * Time.unscaledDeltaTime;
            focus.position += movementMetres * unityUnitsPerMetre;
            streaming?.SnapFocusToTerrain();
        }

        private void SetViewDistance(float value, string band)
        {
            if (adminFpsMode) ExitAdminFpsMode();
            distanceMetres = value;
            ApplyPose();
            Debug.Log($"FIREVIEWER_SPATIAL_VIEW_CHANGED band={band} distanceM={distanceMetres:0}", this);
        }

        private void ApplyPose()
        {
            if (viewerCamera == null || focus == null) return;
            pitchDegrees = Mathf.Clamp(pitchDegrees, MinimumPitchDegrees, MaximumPitchDegrees);
            Quaternion rotation = Quaternion.Euler(pitchDegrees, yawDegrees, 0f);
            viewerCamera.transform.localRotation = rotation;
            viewerCamera.transform.localPosition = rotation * Vector3.back * (distanceMetres * unityUnitsPerMetre);
            KeepCameraAboveTerrain();
        }

        private void KeepCameraAboveTerrain()
        {
            if (streaming == null ||
                !streaming.TrySampleTerrainWorldPoint(viewerCamera.transform.position, out Vector3 terrainPoint))
                return;

            Vector3 up = focus.up.sqrMagnitude > 0.5f ? focus.up.normalized : Vector3.up;
            float clearance = Vector3.Dot(viewerCamera.transform.position - terrainPoint, up);
            float requiredClearance = MinimumTerrainClearanceMetres * unityUnitsPerMetre;
            if (clearance >= requiredClearance) return;
            viewerCamera.transform.position += up * (requiredClearance - clearance);
            Vector3 direction = focus.position - viewerCamera.transform.position;
            if (direction.sqrMagnitude > 0.0001f)
                viewerCamera.transform.rotation = Quaternion.LookRotation(direction.normalized, up);
        }

        private void EnterAdminFpsMode()
        {
            adminFpsMode = true;
            Vector3 euler = viewerCamera.transform.eulerAngles;
            adminFpsPitchDegrees = NormalizeSignedAngle(euler.x);
            adminFpsYawDegrees = NormalizeSignedAngle(euler.y);
            viewerCamera.transform.SetParent(focus.parent, true);
            FollowAdminCameraOnTerrain();
            Debug.Log("FIREVIEWER_SPATIAL_ADMIN_FPS_CHANGED active=true reason=admin_authorized", this);
        }

        private void ExitAdminFpsMode()
        {
            if (!adminFpsMode) return;
            adminFpsMode = false;
            distanceMetres = Mathf.Clamp(
                Vector3.Distance(viewerCamera.transform.position, focus.position) / unityUnitsPerMetre,
                NearDistanceMetres,
                FarDistanceMetres);
            viewerCamera.transform.SetParent(focus, false);
            pitchDegrees = 38f;
            ApplyPose();
            Debug.Log("FIREVIEWER_SPATIAL_ADMIN_FPS_CHANGED active=false", this);
        }

        private void UpdateAdminFpsMode()
        {
            if (!adminEditingAuthorized || !detailNavigationAuthorized)
            {
                ExitAdminFpsMode();
                return;
            }

            if (Input.GetMouseButton(1))
            {
                adminFpsYawDegrees += Input.GetAxis("Mouse X") * adminFpsLookSensitivity * 100f;
                adminFpsPitchDegrees = Mathf.Clamp(
                    adminFpsPitchDegrees - Input.GetAxis("Mouse Y") * adminFpsLookSensitivity * 100f,
                    MinimumFpsPitchDegrees,
                    MaximumFpsPitchDegrees);
            }
            viewerCamera.transform.rotation = Quaternion.Euler(adminFpsPitchDegrees, adminFpsYawDegrees, 0f);

            float horizontal = Input.GetAxisRaw("Horizontal");
            float forward = Input.GetAxisRaw("Vertical");
            float vertical = 0f;
            if (Input.GetKey(KeyCode.E) || Input.GetKey(KeyCode.PageUp)) vertical += 1f;
            if (Input.GetKey(KeyCode.Q) || Input.GetKey(KeyCode.PageDown)) vertical -= 1f;
            Vector3 movement = viewerCamera.transform.right * horizontal + viewerCamera.transform.forward * forward + focus.up * vertical;
            if (movement.sqrMagnitude > 1f) movement.Normalize();
            float boost = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift) ? 4f : 1f;
            viewerCamera.transform.position += movement * (adminFpsSpeedMetresPerSecond * boost * unityUnitsPerMetre * Time.unscaledDeltaTime);
            KeepCameraAboveTerrain();
            FollowAdminCameraOnTerrain();
        }

        private void FollowAdminCameraOnTerrain()
        {
            if (streaming == null ||
                !streaming.TrySampleTerrainWorldPoint(viewerCamera.transform.position, out Vector3 terrainPoint))
                return;
            focus.position = terrainPoint;
        }

        private static float NormalizeSignedAngle(float angle)
        {
            angle %= 360f;
            if (angle > 180f) angle -= 360f;
            if (angle < -180f) angle += 360f;
            return angle;
        }

        private void OnGUI()
        {
            if (!showValidationHud || streaming == null) return;
            FwSpatialStreamingTelemetry state = streaming.Telemetry;
            GUILayout.BeginArea(new Rect(16f, 16f, 360f, 292f), GUI.skin.box);
            GUILayout.Label("FireViewer — contrôle LOD spatial");
            GUILayout.Label($"État : {state.state}    LOD : {state.lod_band}    Caméra : {(adminFpsMode ? "FPS admin" : "orbite")}");
            GUILayout.Label($"Zone : {state.focus_zone}    Distance : {state.view_distance_m:0} m");
            GUILayout.Label($"Altitude sol : {state.focus_elevation_l93_m:0.0} m");
            GUILayout.Label($"Détails : {state.visible_detail_tile_count}/{state.desired_tile_count} tuiles visibles");

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Montmaur")) streaming.FocusMontmaur();
            if (GUILayout.Button("Barsac")) streaming.FocusBarsac();
            if (GUILayout.Button("Ausson")) streaming.FocusAusson();
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("1  Vue globale")) SetFarView();
            GUI.enabled = detailNavigationAuthorized;
            if (GUILayout.Button("2  Moyen")) SetMidView();
            GUI.enabled = detailNavigationAuthorized && streaming?.NearLodDisabled != true;
            if (GUILayout.Button(streaming?.NearLodDisabled == true ? "3  Proche désactivé" : "3  Proche")) SetNearView();
            GUI.enabled = true;
            GUILayout.EndHorizontal();

            GUI.enabled = adminEditingAuthorized && detailNavigationAuthorized;
            if (GUILayout.Button(adminFpsMode ? "4  Quitter FPS administration" : "4  FPS administration — édition/placement"))
                ToggleAdminFpsMode();
            GUI.enabled = true;

            GUILayout.Label(detailNavigationAuthorized
                ? (adminFpsMode
                    ? "FPS admin : souris droite, ZQSD, Q/E, Maj accélère"
                    : "Souris droite : orbite   Molette : zoom")
                : "Connexion requise pour zoom, vue moyenne et vue proche");
            GUILayout.Label(adminEditingAuthorized
                ? "Le mode FPS est réservé au rôle administrateur"
                : "Administration requise pour la caméra FPS");
            if (!adminFpsMode) GUILayout.Label("ZQSD/flèches : déplacer sur le relief");
            if (!string.IsNullOrEmpty(state.last_error)) GUILayout.Label($"Erreur : {state.last_error}");
            GUILayout.EndArea();
        }

        private bool RequireDetailNavigation(string requestedBand)
        {
            if (detailNavigationAuthorized) return true;
            Debug.LogWarning($"FIREVIEWER_SPATIAL_NAVIGATION_DENIED band={requestedBand} reason=authentication_required", this);
            return false;
        }

        private bool RequireAdminEditing()
        {
            if (adminEditingAuthorized && detailNavigationAuthorized) return true;
            Debug.LogWarning("FIREVIEWER_SPATIAL_ADMIN_FPS_DENIED reason=admin_authentication_required", this);
            return false;
        }
    }
}
