using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace FireViewer.SpatialTiles
{
    [Serializable]
    public sealed class FwSpatialStreamingTelemetry
    {
        public string state = "disabled";
        public string last_error = string.Empty;
        public string focus_zone = string.Empty;
        public string lod_band = "far";
        public double view_distance_m;
        public double focus_easting_l93_m;
        public double focus_northing_l93_m;
        public double focus_elevation_l93_m;
        public int desired_tile_count;
        public int resident_tile_count;
        public int visible_detail_tile_count;
        public int in_flight_tile_count;
        public int maximum_resident_tile_count;
        public int revision;
    }

    [DisallowMultipleComponent]
    [AddComponentMenu("FireViewer/Spatial Tile Streaming Controller")]
    public sealed class FwSpatialTileStreamingController : MonoBehaviour
    {
        private static readonly int DetailBoundsCountProperty = Shader.PropertyToID("_FwDetailBoundsCount");
        private static readonly int DetailBoundsProperty = Shader.PropertyToID("_FwDetailBounds");
        private readonly Vector4[] detailBounds = new Vector4[FwSpatialLodPlanner.AbsoluteMaximumResidentTiles];
        [SerializeField] private string catalogUrl = string.Empty;
        [SerializeField] private Transform focus;
        [SerializeField] private Camera viewerCamera;
        [SerializeField] private Transform localFrameOrigin;
        [SerializeField] private Transform contentRoot;
        [SerializeField, Min(0.0001f)] private float unityUnitsPerMetre = 1f;
        [SerializeField, Range(1, FwSpatialLodPlanner.AbsoluteMaximumResidentTiles)] private int runtimeResidentBudget = 16;
        [SerializeField, Min(0.05f)] private float evaluationIntervalSeconds = 0.25f;
        [SerializeField, Min(0f)] private float nearVisibleTileMarginMetres = 50f;
        [SerializeField, Min(0f)] private float nearTerrainVerticalMarginMetres = 60f;
        [SerializeField] private bool detailStreamingAuthorized;
        [SerializeField] private bool startAutomatically = true;
        [SerializeField] private FwSpatialStreamingTelemetry telemetry = new();

        private readonly Dictionary<string, FwLoadedTile> residents = new(StringComparer.Ordinal);
        private readonly Dictionary<string, FwRegularTerrainSampler> residentTerrainSamplers = new(StringComparer.Ordinal);
        private readonly FwAtomicPublicationState publication = new();
        private FwRemoteCatalog catalog;
        private FwLoadedTile far;
        private FwRegularTerrainSampler farTerrainSampler;
        private Coroutine bootstrapCoroutine;
        private Coroutine reconcileCoroutine;
        private FwTileSelectionPlan targetPlan;
        private string targetSignature = string.Empty;
        private string pendingFocusZone = string.Empty;
        private int revision;
        private float nextEvaluation;
        private bool ready;
        private bool stopping;

        public FwRemoteCatalog Catalog => catalog;
        public FwSpatialStreamingTelemetry Telemetry => telemetry;
        public bool IsReady => ready;
        public int ResidentTileCount => residents.Count;
        public bool DetailStreamingAuthorized => detailStreamingAuthorized;
        public bool DetailStreamingEnabled => catalog?.lod_policy?.detail?.enabled ?? false;
        public bool NearLodDisabled => catalog?.lod_policy?.detail?.near_disabled ?? false;
        public Camera ViewerCamera => viewerCamera != null ? viewerCamera : Camera.main;

        public void Configure(
            string remoteCatalogUrl,
            Transform focusTransform,
            Transform frameOrigin,
            Transform streamedContentRoot = null,
            float unitsPerMetre = 1f,
            int residentBudget = 16,
            bool authorizeDetailStreaming = false)
        {
            catalogUrl = remoteCatalogUrl ?? string.Empty;
            focus = focusTransform;
            localFrameOrigin = frameOrigin;
            contentRoot = streamedContentRoot;
            unityUnitsPerMetre = unitsPerMetre;
            runtimeResidentBudget = Mathf.Clamp(residentBudget, 1, FwSpatialLodPlanner.AbsoluteMaximumResidentTiles);
            detailStreamingAuthorized = authorizeDetailStreaming;
            if (Application.isPlaying && isActiveAndEnabled) StartStreaming();
        }

        public void SetViewerCamera(Camera camera) => viewerCamera = camera;

        /// <summary>
        /// The authenticated host owns this decision.  When access is
        /// revoked, the streamer itself returns to FAR so moving the camera
        /// from another script cannot download MID/NEAR assets.
        /// </summary>
        public void SetDetailStreamingAuthorized(bool authorized)
        {
            detailStreamingAuthorized = authorized;
            targetSignature = string.Empty;
            if (ready && !stopping) Evaluate(true);
        }

        public void StartStreaming()
        {
            StopStreaming();
            stopping = false;
            telemetry.state = "loading_catalog";
            telemetry.last_error = string.Empty;
            bootstrapCoroutine = StartCoroutine(Bootstrap());
        }

        public void StopStreaming()
        {
            stopping = true;
            ready = false;
            if (bootstrapCoroutine != null) StopCoroutine(bootstrapCoroutine);
            if (reconcileCoroutine != null) StopCoroutine(reconcileCoroutine);
            bootstrapCoroutine = null;
            reconcileCoroutine = null;
            ReleaseAllDetails();
            far?.Dispose();
            far = null;
            farTerrainSampler = null;
            residentTerrainSamplers.Clear();
            ClearFarDetailMask();
            targetPlan = null;
            targetSignature = string.Empty;
            telemetry.state = "stopped";
            RefreshTelemetry();
        }

        public bool FocusZone(string zoneId)
        {
            if (!FwKnownFocusZones.TryGet(zoneId, out FwFocusZone zone)) return false;
            telemetry.focus_zone = zone.Id;
            if (catalog == null)
            {
                pendingFocusZone = zone.Id;
                return true;
            }
            SetFocusLambert(zone.Easting, zone.Northing, zone.Elevation);
            return true;
        }

        public void FocusMontmaur() => FocusZone(FwKnownFocusZones.Montmaur.Id);
        public void FocusBarsac() => FocusZone(FwKnownFocusZones.Barsac.Id);
        public void FocusAusson() => FocusZone(FwKnownFocusZones.Ausson.Id);

        public void SetFocusLambert(double easting, double northing)
        {
            SetFocusLambert(easting, northing, null);
        }

        public void SetFocusLambert(double easting, double northing, double? elevation)
        {
            if (catalog == null) throw new InvalidOperationException("Catalog must be loaded before setting an arbitrary Lambert-93 focus.");
            EnsureFocusTransform();
            float localX = checked((float)((easting - catalog.origin_l93_m[0]) * unityUnitsPerMetre));
            float localZ = checked((float)((northing - catalog.origin_l93_m[1]) * unityUnitsPerMetre));
            Vector3 currentLocal = localFrameOrigin != null ? localFrameOrigin.InverseTransformPoint(focus.position) : focus.position;
            float localY = elevation.HasValue
                ? checked((float)((elevation.Value - catalog.origin_l93_m[2]) * unityUnitsPerMetre))
                : currentLocal.y;
            if (TrySampleTerrainHeightMetres(easting, northing, out float groundUpMetres))
                localY = groundUpMetres * unityUnitsPerMetre;
            Vector3 local = new(localX, localY, localZ);
            focus.position = localFrameOrigin != null ? localFrameOrigin.TransformPoint(local) : local;
            telemetry.focus_easting_l93_m = easting;
            telemetry.focus_northing_l93_m = northing;
            telemetry.focus_elevation_l93_m = catalog.origin_l93_m[2] + localY / unityUnitsPerMetre;
            Evaluate(true);
        }

        public bool TryGetFocusLambert(out double easting, out double northing)
        {
            easting = 0d; northing = 0d;
            if (catalog == null || focus == null || unityUnitsPerMetre <= 0f) return false;
            Vector3 local = localFrameOrigin != null ? localFrameOrigin.InverseTransformPoint(focus.position) : focus.position;
            easting = catalog.origin_l93_m[0] + local.x / unityUnitsPerMetre;
            northing = catalog.origin_l93_m[1] + local.z / unityUnitsPerMetre;
            return true;
        }

        public bool SnapFocusToTerrain()
        {
            if (catalog == null || focus == null || unityUnitsPerMetre <= 0f ||
                !TryGetFocusLambert(out double easting, out double northing) ||
                !TrySampleTerrainHeightMetres(easting, northing, out float groundUpMetres))
                return false;

            Vector3 local = localFrameOrigin != null ? localFrameOrigin.InverseTransformPoint(focus.position) : focus.position;
            local.y = groundUpMetres * unityUnitsPerMetre;
            focus.position = localFrameOrigin != null ? localFrameOrigin.TransformPoint(local) : local;
            telemetry.focus_elevation_l93_m = catalog.origin_l93_m[2] + groundUpMetres;
            return true;
        }

        public bool TrySampleTerrainWorldPoint(Vector3 worldPosition, out Vector3 surfaceWorldPoint)
        {
            surfaceWorldPoint = default;
            if (catalog == null || unityUnitsPerMetre <= 0f) return false;
            Vector3 local = localFrameOrigin != null ? localFrameOrigin.InverseTransformPoint(worldPosition) : worldPosition;
            double easting = catalog.origin_l93_m[0] + local.x / unityUnitsPerMetre;
            double northing = catalog.origin_l93_m[1] + local.z / unityUnitsPerMetre;
            if (!TrySampleTerrainHeightMetres(easting, northing, out float groundUpMetres)) return false;
            local.y = groundUpMetres * unityUnitsPerMetre;
            surfaceWorldPoint = localFrameOrigin != null ? localFrameOrigin.TransformPoint(local) : local;
            return true;
        }

        private void Start()
        {
            if (startAutomatically && !string.IsNullOrWhiteSpace(catalogUrl)) StartStreaming();
        }

        private void Update()
        {
            if (!ready || stopping || Time.unscaledTime < nextEvaluation) return;
            nextEvaluation = Time.unscaledTime + evaluationIntervalSeconds;
            Evaluate(false);
        }

        private void OnDestroy() => StopStreaming();

        private IEnumerator Bootstrap()
        {
            FwRemoteCatalog loadedCatalog = null;
            string error = null;
            yield return FwRemoteTileClient.LoadCatalog(catalogUrl, value => loadedCatalog = value, value => error = value);
            if (stopping) yield break;
            if (error != null) { Fail("catalog_load_failed", error); yield break; }
            catalog = loadedCatalog;
            if (!catalog.lod_policy.detail.enabled)
                detailStreamingAuthorized = false;
            telemetry.maximum_resident_tile_count = Math.Min(runtimeResidentBudget, catalog.lod_policy.detail.maximum_resident_tile_count);
            telemetry.state = "loading_far";
            FwLoadedTile loadedFar = null;
            yield return FwRemoteTileClient.LoadFarTile(catalogUrl, catalog, ContentParent, unityUnitsPerMetre, value => loadedFar = value, value => error = value);
            if (stopping) { loadedFar?.Dispose(); yield break; }
            if (error != null) { Fail("far_load_failed", error); yield break; }
            far = loadedFar;
            farTerrainSampler = new FwRegularTerrainSampler(far.Geometry.Terrain);
            far.Root.SetActive(true);
            ClearFarDetailMask();
            ready = true;
            telemetry.state = "far";
            Debug.Log(
                $"FIREVIEWER_SPATIAL_FAR_READY vertices={far.Geometry.Terrain.Vertices.Length} " +
                $"triangles={far.Geometry.Terrain.Triangles.Length / 3} catalogTiles={catalog.tiles.Length}",
                this);
            bootstrapCoroutine = null;
            if (!string.IsNullOrEmpty(pendingFocusZone))
            {
                string zone = pendingFocusZone;
                pendingFocusZone = string.Empty;
                FocusZone(zone);
            }
            else
            {
                EnsureFocusTransform();
                Evaluate(true);
            }
        }

        private void Evaluate(bool force)
        {
            SnapFocusToTerrain();
            if (!ready || !TryGetFocusLambert(out double east, out double north)) return;
            if (!catalog.lod_policy.detail.enabled)
            {
                ShowFarAndHideDetails();
                telemetry.focus_easting_l93_m = east;
                telemetry.focus_northing_l93_m = north;
                telemetry.view_distance_m = double.PositiveInfinity;
                telemetry.lod_band = "global";
                telemetry.desired_tile_count = 0;
                telemetry.state = "global";
                telemetry.last_error = string.Empty;
                RefreshTelemetry();
                return;
            }
            double viewDistance = detailStreamingAuthorized ? ViewDistanceMetres() : double.PositiveInfinity;
            string lodBand = FwSpatialLodPlanner.ClassifyBand(
                viewDistance,
                catalog.lod_policy.detail.near_disabled);
            FwPlanarFootprint visibleFootprint = null;
            bool usesDetailTiles = !string.Equals(lodBand, "far", StringComparison.Ordinal);
            if (usesDetailTiles && !TryBuildCameraTerrainFootprint(viewDistance, out visibleFootprint))
            {
                // A raised camera can stop intersecting the terrain envelope.
                // Never retain the previous detail mask in that state: it
                // would keep punching obsolete rectangles out of the global
                // MNT and make distant terrain appear to disappear.  Publish
                // an explicit FAR-only fallback until the camera sees a valid
                // terrain footprint again.
                var fallback = new FwTileSelectionPlan(
                    Array.Empty<FwCatalogTile>(),
                    "Detail camera footprint is unavailable; FAR fallback is active.");
                QueuePlan(fallback, east, north, viewDistance, lodBand, force);
                return;
            }
            FwTileSelectionPlan plan = FwSpatialLodPlanner.Select(
                catalog,
                east,
                north,
                viewDistance,
                runtimeResidentBudget,
                visibleFootprint);
            QueuePlan(plan, east, north, viewDistance, lodBand, force);
        }

        private void QueuePlan(
            FwTileSelectionPlan plan,
            double east,
            double north,
            double viewDistance,
            string lodBand,
            bool force)
        {
            string signature = PlanSignature(plan);
            telemetry.focus_easting_l93_m = east;
            telemetry.focus_northing_l93_m = north;
            telemetry.view_distance_m = viewDistance;
            telemetry.lod_band = lodBand;
            if (!force && string.Equals(signature, targetSignature, StringComparison.Ordinal)) return;
            targetPlan = plan;
            targetSignature = signature;
            revision++;
            telemetry.revision = revision;
            telemetry.desired_tile_count = plan.Tiles.Length;
            if (plan.IsBlocked)
            {
                Debug.LogWarning(
                    $"FIREVIEWER_SPATIAL_PLAN_BLOCKED zone={telemetry.focus_zone} band={lodBand} " +
                    $"distanceM={viewDistance:0} reason={plan.BlockingError}",
                    this);
            }
            ScheduleReconcile();
        }

        private void ScheduleReconcile()
        {
            if (!ready || stopping || reconcileCoroutine != null) return;
            reconcileCoroutine = StartCoroutine(Reconcile());
        }

        private IEnumerator Reconcile()
        {
            yield return null;
            int workingRevision = revision;
            FwTileSelectionPlan plan = targetPlan;
            HashSet<string> previouslyVisible = VisibleResidentIds();
            var newlyLoaded = new HashSet<string>(StringComparer.Ordinal);
            var desiredIds = new HashSet<string>(TileIds(plan.Tiles), StringComparer.Ordinal);
            publication.Begin(TileIds(plan.Tiles));
            foreach (FwCatalogTile tile in plan.Tiles)
                if (residents.ContainsKey(tile.id)) publication.Stage(tile.id);

            // Keep a global, non-drawn fallback resident in memory.  It is
            // activated only while a different detail footprint is staged,
            // and masked beneath any previously visible detail tiles.
            if (!desiredIds.SetEquals(previouslyVisible) && far?.Root != null)
            {
                far.Root.SetActive(true);
                PublishFarDetailMaskForVisibleResidents();
            }

            if (plan.IsBlocked)
            {
                publication.Fail(plan.BlockingError);
                // A blocked or geometrically unavailable NEAR plan must not
                // leave an obsolete FAR cut-out on screen.  Keep cached detail
                // objects resident, but hide them and clear their mask so the
                // complete global MNT is immediately visible.
                ShowFarAndHideDetails();
                telemetry.state = "far_detail_fallback";
                telemetry.last_error = plan.BlockingError;
                reconcileCoroutine = null;
                RefreshTelemetry();
                yield break;
            }
            if (plan.Tiles.Length == 0)
            {
                ShowFarAndHideDetails();
                ReleaseAllDetails();
                telemetry.state = "far";
                telemetry.last_error = string.Empty;
                reconcileCoroutine = null;
                RefreshTelemetry();
                yield break;
            }

            telemetry.state = "loading_complete_detail_neighbourhood";
            telemetry.last_error = string.Empty;
            foreach (FwCatalogTile tile in plan.Tiles)
            {
                if (workingRevision != revision) break;
                if (residents.ContainsKey(tile.id)) continue;
                if (residents.Count >= telemetry.maximum_resident_tile_count)
                {
                    if (!ReleaseOneOutside(desiredIds))
                    {
                        publication.Fail("Global resident tile budget reached before atomic publication.");
                        break;
                    }
                    PublishFarDetailMaskForVisibleResidents();
                }
                telemetry.in_flight_tile_count = 1;
                FwLoadedTile loaded = null;
                string error = null;
                yield return FwRemoteTileClient.LoadDetailTile(catalogUrl, catalog, tile, ContentParent, unityUnitsPerMetre, value => loaded = value, value => error = value);
                telemetry.in_flight_tile_count = 0;
                if (workingRevision != revision)
                {
                    loaded?.Dispose();
                    break;
                }
                if (error != null || loaded == null)
                {
                    publication.Fail(error ?? $"Tile {tile.id} returned no object.");
                    break;
                }
                loaded.Root.SetActive(false);
                residents.Add(tile.id, loaded);
                residentTerrainSamplers[tile.id] = new FwRegularTerrainSampler(loaded.Geometry.Terrain);
                newlyLoaded.Add(tile.id);
                publication.Stage(tile.id);
                RefreshTelemetry();
            }

            if (workingRevision != revision)
            {
                RestorePreviousPublication(previouslyVisible, newlyLoaded);
                reconcileCoroutine = null;
                ScheduleReconcile();
                yield break;
            }
            if (!publication.TryPublish())
            {
                string failure = publication.Failure.Length > 0 ? publication.Failure : "Complete detail neighbourhood is not resident.";
                publication.Fail(failure);
                RestorePreviousPublication(previouslyVisible, newlyLoaded);
                telemetry.state = VisibleResidentIds().Count > 0 ? "detail_previous_retained" : "far_detail_load_failed";
                telemetry.last_error = failure;
                // A transient HTTP outage must not pin the runtime forever to
                // the fallback.  Clear the signature so the normal evaluator
                // retries the same complete neighbourhood after a short delay.
                targetSignature = string.Empty;
                nextEvaluation = Time.unscaledTime + 2f;
                reconcileCoroutine = null;
                RefreshTelemetry();
                yield break;
            }

            foreach (FwCatalogTile tile in plan.Tiles) residents[tile.id].Root.SetActive(true);
            ReleaseOutside(plan.Tiles);
            // Preserve the global MNT behind MID/NEAR so the horizon never
            // disappears.  The terrain shader clips FAR only beneath the
            // exact detailed tile rectangles, preventing overlap/z-fighting.
            PublishFarDetailMask(plan.Tiles);
            if (far?.Root != null) far.Root.SetActive(true);
            telemetry.state = "detail_atomic";
            telemetry.last_error = string.Empty;
            reconcileCoroutine = null;
            RefreshTelemetry();
            int treeCount = 0;
            int buildingMeshCount = 0;
            int roadMeshCount = 0;
            int waterMeshCount = 0;
            foreach (FwCatalogTile tile in plan.Tiles)
            {
                FwTileGeometry geometry = residents[tile.id].Geometry;
                treeCount += geometry.Trees?.Length ?? 0;
                buildingMeshCount += geometry.Buildings?.Length ?? 0;
                roadMeshCount += geometry.Roads?.Length ?? 0;
                waterMeshCount += geometry.Water?.Length ?? 0;
            }
            Debug.Log(
                $"FIREVIEWER_SPATIAL_DETAIL_READY zone={telemetry.focus_zone} band={telemetry.lod_band} " +
                $"tiles={plan.Tiles.Length} trees={treeCount} buildingMeshes={buildingMeshCount} " +
                $"roadMeshes={roadMeshCount} waterMeshes={waterMeshCount}",
                this);
        }

        private void ShowFarAndHideDetails()
        {
            if (far?.Root != null) far.Root.SetActive(true);
            ClearFarDetailMask();
            foreach (FwLoadedTile tile in residents.Values) if (tile.Root != null) tile.Root.SetActive(false);
        }

        private void ReleaseOutside(FwCatalogTile[] desired)
        {
            var keep = new HashSet<string>(StringComparer.Ordinal);
            foreach (FwCatalogTile tile in desired) keep.Add(tile.id);
            var release = new List<string>();
            foreach (string id in residents.Keys) if (!keep.Contains(id)) release.Add(id);
            foreach (string id in release) Release(id);
        }

        private void Release(string tileId)
        {
            if (!residents.TryGetValue(tileId, out FwLoadedTile tile)) return;
            residents.Remove(tileId);
            residentTerrainSamplers.Remove(tileId);
            tile.Dispose();
        }

        private void ReleaseAllDetails()
        {
            var ids = new List<string>(residents.Keys);
            foreach (string id in ids) Release(id);
        }

        private HashSet<string> VisibleResidentIds()
        {
            var visible = new HashSet<string>(StringComparer.Ordinal);
            foreach (KeyValuePair<string, FwLoadedTile> pair in residents)
                if (pair.Value.Root != null && pair.Value.Root.activeSelf) visible.Add(pair.Key);
            return visible;
        }

        private bool ReleaseOneOutside(HashSet<string> desiredIds)
        {
            foreach (string id in new List<string>(residents.Keys))
            {
                if (desiredIds.Contains(id)) continue;
                Release(id);
                return true;
            }
            return false;
        }

        private void RestorePreviousPublication(HashSet<string> previousIds, HashSet<string> newlyLoaded)
        {
            foreach (string id in newlyLoaded) Release(id);
            foreach (KeyValuePair<string, FwLoadedTile> pair in residents)
                if (pair.Value.Root != null) pair.Value.Root.SetActive(previousIds.Contains(pair.Key));
            if (far?.Root != null) far.Root.SetActive(true);
            PublishFarDetailMaskForVisibleResidents();
            RefreshTelemetry();
        }

        private void PublishFarDetailMaskForVisibleResidents()
        {
            if (catalog?.tiles == null)
            {
                ClearFarDetailMask();
                return;
            }
            HashSet<string> visible = VisibleResidentIds();
            if (visible.Count == 0)
            {
                ClearFarDetailMask();
                return;
            }
            var tiles = new List<FwCatalogTile>(visible.Count);
            foreach (FwCatalogTile tile in catalog.tiles)
                if (visible.Contains(tile.id)) tiles.Add(tile);
            PublishFarDetailMask(tiles.ToArray());
        }

        private void EnsureFocusTransform()
        {
            if (focus != null) return;
            var target = new GameObject("FireViewer Focus Target");
            focus = target.transform;
            if (localFrameOrigin != null) focus.SetParent(localFrameOrigin, false);
        }

        private void RefreshTelemetry()
        {
            int visible = 0;
            foreach (FwLoadedTile tile in residents.Values) if (tile.Root != null && tile.Root.activeSelf) visible++;
            telemetry.resident_tile_count = residents.Count;
            telemetry.visible_detail_tile_count = visible;
        }

        private void Fail(string code, string detail)
        {
            ready = false;
            ClearFarDetailMask();
            telemetry.state = "error";
            telemetry.last_error = $"{code}: {detail}";
            Debug.LogError($"FIREVIEWER_SPATIAL_STREAMING_FAILED {telemetry.last_error}", this);
        }

        private Transform ContentParent => contentRoot != null ? contentRoot : transform;

        private double ViewDistanceMetres()
        {
            if (focus == null || unityUnitsPerMetre <= 0f) return double.PositiveInfinity;
            Camera camera = viewerCamera != null ? viewerCamera : Camera.main;
            if (camera != null && camera.transform != focus)
                return Vector3.Distance(camera.transform.position, focus.position) / unityUnitsPerMetre;
            Vector3 local = localFrameOrigin != null ? localFrameOrigin.InverseTransformPoint(focus.position) : focus.position;
            return Math.Abs(local.y) / unityUnitsPerMetre;
        }

        private bool TryBuildCameraTerrainFootprint(double viewDistanceMetres, out FwPlanarFootprint footprint)
        {
            footprint = null;
            Camera camera = viewerCamera != null ? viewerCamera : Camera.main;
            if (camera == null || focus == null || catalog?.lod_policy?.detail == null || unityUnitsPerMetre <= 0f) return false;

            float visibleDepthMetres = checked((float)(
                viewDistanceMetres + catalog.lod_policy.detail.preload_radius_m));
            float farDistance = Mathf.Min(
                camera.farClipPlane,
                visibleDepthMetres * unityUnitsPerMetre);
            if (farDistance <= camera.nearClipPlane) return false;
            var nearCorners = new Vector3[4];
            var farCorners = new Vector3[4];
            camera.CalculateFrustumCorners(new Rect(0f, 0f, 1f, 1f), camera.nearClipPlane, Camera.MonoOrStereoscopicEye.Mono, nearCorners);
            camera.CalculateFrustumCorners(new Rect(0f, 0f, 1f, 1f), farDistance, Camera.MonoOrStereoscopicEye.Mono, farCorners);

            var worldCorners = new Vector3[8];
            for (int index = 0; index < 4; index++)
            {
                worldCorners[index] = camera.transform.TransformPoint(nearCorners[index]);
                worldCorners[index + 4] = camera.transform.TransformPoint(farCorners[index]);
            }
            int[,] edges =
            {
                { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
                { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
                { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 },
            };
            Vector3 planeNormal = localFrameOrigin != null ? localFrameOrigin.up : Vector3.up;
            Vector3 focusLocal = localFrameOrigin != null ? localFrameOrigin.InverseTransformPoint(focus.position) : focus.position;
            float minimumTerrainUpMetres = focusLocal.y / unityUnitsPerMetre;
            float maximumTerrainUpMetres = minimumTerrainUpMetres;
            float minimumEastMetres = float.PositiveInfinity;
            float minimumNorthMetres = float.PositiveInfinity;
            float maximumEastMetres = float.NegativeInfinity;
            float maximumNorthMetres = float.NegativeInfinity;
            foreach (Vector3 worldCorner in worldCorners)
            {
                Vector3 localCorner = localFrameOrigin != null
                    ? localFrameOrigin.InverseTransformPoint(worldCorner)
                    : worldCorner;
                float eastMetres = localCorner.x / unityUnitsPerMetre;
                float northMetres = localCorner.z / unityUnitsPerMetre;
                minimumEastMetres = Mathf.Min(minimumEastMetres, eastMetres);
                minimumNorthMetres = Mathf.Min(minimumNorthMetres, northMetres);
                maximumEastMetres = Mathf.Max(maximumEastMetres, eastMetres);
                maximumNorthMetres = Mathf.Max(maximumNorthMetres, northMetres);
            }
            float rangeMarginMetres = nearVisibleTileMarginMetres;
            bool sampledTerrainRange = farTerrainSampler != null && farTerrainSampler.TryGetHeightRange(
                minimumEastMetres - rangeMarginMetres,
                minimumNorthMetres - rangeMarginMetres,
                maximumEastMetres + rangeMarginMetres,
                maximumNorthMetres + rangeMarginMetres,
                out minimumTerrainUpMetres,
                out maximumTerrainUpMetres);
            if (!sampledTerrainRange)
            {
                // TryGetHeightRange resets its outputs when the projected
                // frustum lies outside the AOI.  Restore the real focus height
                // explicitly instead of constructing a slab around project Y=0.
                minimumTerrainUpMetres = focusLocal.y / unityUnitsPerMetre;
                maximumTerrainUpMetres = minimumTerrainUpMetres;
            }
            minimumTerrainUpMetres -= nearTerrainVerticalMarginMetres;
            maximumTerrainUpMetres += nearTerrainVerticalMarginMetres;
            Vector3 lowerLocal = new(focusLocal.x, minimumTerrainUpMetres * unityUnitsPerMetre, focusLocal.z);
            Vector3 upperLocal = new(focusLocal.x, maximumTerrainUpMetres * unityUnitsPerMetre, focusLocal.z);
            Vector3 lowerPlanePoint = localFrameOrigin != null ? localFrameOrigin.TransformPoint(lowerLocal) : lowerLocal;
            Vector3 upperPlanePoint = localFrameOrigin != null ? localFrameOrigin.TransformPoint(upperLocal) : upperLocal;
            var intersections = new List<FwPlanarPoint>(12);
            const float planeEpsilon = 0.001f;
            foreach (Vector3 corner in worldCorners)
            {
                float lowerDistance = Vector3.Dot(planeNormal, corner - lowerPlanePoint);
                float upperDistance = Vector3.Dot(planeNormal, corner - upperPlanePoint);
                if (lowerDistance >= -planeEpsilon && upperDistance <= planeEpsilon)
                    AddLambertFootprintPoint(intersections, corner);
            }
            for (int edge = 0; edge < edges.GetLength(0); edge++)
            {
                Vector3 start = worldCorners[edges[edge, 0]];
                Vector3 end = worldCorners[edges[edge, 1]];
                AddPlaneIntersection(intersections, start, end, planeNormal, lowerPlanePoint, planeEpsilon);
                AddPlaneIntersection(intersections, start, end, planeNormal, upperPlanePoint, planeEpsilon);
            }

            if (intersections.Count < 3) return false;
            try
            {
                footprint = new FwPlanarFootprint(intersections.ToArray(), nearVisibleTileMarginMetres);
                return true;
            }
            catch (ArgumentException)
            {
                return false;
            }
        }

        private void AddPlaneIntersection(
            List<FwPlanarPoint> intersections,
            Vector3 start,
            Vector3 end,
            Vector3 planeNormal,
            Vector3 planePoint,
            float epsilon)
        {
            float startDistance = Vector3.Dot(planeNormal, start - planePoint);
            float endDistance = Vector3.Dot(planeNormal, end - planePoint);
            if (Mathf.Abs(startDistance) <= epsilon) AddLambertFootprintPoint(intersections, start);
            if (Mathf.Abs(endDistance) <= epsilon) AddLambertFootprintPoint(intersections, end);
            if ((startDistance < -epsilon && endDistance > epsilon) ||
                (startDistance > epsilon && endDistance < -epsilon))
            {
                float factor = startDistance / (startDistance - endDistance);
                AddLambertFootprintPoint(intersections, Vector3.LerpUnclamped(start, end, factor));
            }
        }

        private bool TrySampleTerrainHeightMetres(double easting, double northing, out float groundUpMetres)
        {
            groundUpMetres = 0f;
            if (catalog == null) return false;
            float localEast = checked((float)(easting - catalog.origin_l93_m[0]));
            float localNorth = checked((float)(northing - catalog.origin_l93_m[1]));
            foreach (FwRegularTerrainSampler sampler in residentTerrainSamplers.Values)
                if (sampler.TrySample(localEast, localNorth, out groundUpMetres)) return true;
            return farTerrainSampler != null && farTerrainSampler.TrySample(localEast, localNorth, out groundUpMetres);
        }

        private void AddLambertFootprintPoint(List<FwPlanarPoint> points, Vector3 worldPoint)
        {
            Vector3 local = localFrameOrigin != null ? localFrameOrigin.InverseTransformPoint(worldPoint) : worldPoint;
            var point = new FwPlanarPoint(
                catalog.origin_l93_m[0] + local.x / unityUnitsPerMetre,
                catalog.origin_l93_m[1] + local.z / unityUnitsPerMetre);
            foreach (FwPlanarPoint existing in points)
            {
                double deltaEast = existing.East - point.East;
                double deltaNorth = existing.North - point.North;
                if (deltaEast * deltaEast + deltaNorth * deltaNorth <= 0.01d) return;
            }
            points.Add(point);
        }

        private static IEnumerable<string> TileIds(FwCatalogTile[] tiles)
        {
            foreach (FwCatalogTile tile in tiles) yield return tile.id;
        }

        private static string PlanSignature(FwTileSelectionPlan plan)
        {
            if (plan.IsBlocked) return $"blocked:{plan.BlockingError}";
            if (plan.Tiles.Length == 0) return "far";
            var ids = new string[plan.Tiles.Length];
            for (int index = 0; index < ids.Length; index++) ids[index] = plan.Tiles[index].id;
            return string.Join("|", ids);
        }

        private void PublishFarDetailMask(FwCatalogTile[] tiles)
        {
            if (catalog?.origin_l93_m == null || tiles == null)
            {
                ClearFarDetailMask();
                return;
            }
            int count = Mathf.Min(tiles.Length, detailBounds.Length);
            for (int index = 0; index < count; index++)
            {
                double[] bounds = tiles[index].bounds_l93_m;
                detailBounds[index] = new Vector4(
                    checked((float)(bounds[0] - catalog.origin_l93_m[0])),
                    checked((float)(bounds[1] - catalog.origin_l93_m[1])),
                    checked((float)(bounds[2] - catalog.origin_l93_m[0])),
                    checked((float)(bounds[3] - catalog.origin_l93_m[1])));
            }
            for (int index = count; index < detailBounds.Length; index++) detailBounds[index] = Vector4.zero;
            Shader.SetGlobalVectorArray(DetailBoundsProperty, detailBounds);
            Shader.SetGlobalInt(DetailBoundsCountProperty, count);
        }

        private void ClearFarDetailMask()
        {
            Shader.SetGlobalInt(DetailBoundsCountProperty, 0);
        }
    }
}
