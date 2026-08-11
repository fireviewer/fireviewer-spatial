using System;
using System.Collections;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace FireViewer.SpatialTiles
{
    public sealed class FwLoadedTile : IDisposable
    {
        private readonly FwTileUnityBuilder builder;

        internal FwLoadedTile(GameObject root, Texture2D imagery, FwTileGeometry geometry, FwTileUnityBuilder builder)
        {
            Root = root;
            Imagery = imagery;
            Geometry = geometry;
            this.builder = builder;
        }

        public GameObject Root { get; }
        public Texture2D Imagery { get; }
        public FwTileGeometry Geometry { get; }

        public void Dispose()
        {
            if (Root != null) DestroyOwned(Root);
            builder?.Dispose();
            if (Imagery != null) DestroyOwned(Imagery);
        }

        private static void DestroyOwned(UnityEngine.Object value)
        {
            if (Application.isPlaying) UnityEngine.Object.Destroy(value);
            else UnityEngine.Object.DestroyImmediate(value);
        }
    }

    /// <summary>
    /// Direct HTTP bridge for the catalog produced by export_remote_catalog.py.
    /// It intentionally does not use Addressables: the immutable .fwtile and
    /// orthophoto references in the business catalog are the runtime assets.
    /// </summary>
    public static class FwRemoteTileClient
    {
        public static IEnumerator LoadCatalog(
            string catalogUrl,
            Action<FwRemoteCatalog> succeeded,
            Action<string> failed)
        {
            if (!Uri.TryCreate(catalogUrl, UriKind.Absolute, out _))
            {
                failed?.Invoke("Catalog URL must be absolute.");
                yield break;
            }
            byte[] bytes = null;
            string error = null;
            yield return Download(catalogUrl, null, 0, (value) => bytes = value, (value) => error = value);
            if (error != null)
            {
                failed?.Invoke(error);
                yield break;
            }
            try
            {
                FwRemoteCatalog catalog = new FwUnityJsonParser().Parse<FwRemoteCatalog>(Encoding.UTF8.GetString(bytes));
                catalog.Validate();
                succeeded?.Invoke(catalog);
            }
            catch (Exception exception)
            {
                failed?.Invoke($"Catalog validation failed: {exception.Message}");
            }
        }

        public static IEnumerator LoadDetailTile(
            string catalogUrl,
            FwRemoteCatalog catalog,
            FwCatalogTile tile,
            Transform parent,
            float unityUnitsPerMetre,
            Action<FwLoadedTile> succeeded,
            Action<string> failed)
        {
            if (catalog == null || tile == null)
            {
                failed?.Invoke("Catalog or tile is missing.");
                yield break;
            }
            yield return LoadTile(
                catalogUrl,
                catalog.origin_l93_m,
                tile.payload,
                tile.imagery,
                parent,
                unityUnitsPerMetre,
                tile.id,
                succeeded,
                failed);
        }

        public static IEnumerator LoadFarTile(
            string catalogUrl,
            FwRemoteCatalog catalog,
            Transform parent,
            float unityUnitsPerMetre,
            Action<FwLoadedTile> succeeded,
            Action<string> failed)
        {
            if (catalog?.lod_policy?.far == null)
            {
                failed?.Invoke("Catalog far LOD is missing.");
                yield break;
            }
            yield return LoadTile(
                catalogUrl,
                catalog.origin_l93_m,
                catalog.lod_policy.far.terrain,
                catalog.lod_policy.far.imagery,
                parent,
                unityUnitsPerMetre,
                "global-far",
                succeeded,
                failed);
        }

        public static bool TryResolveAssetUrl(string catalogUrl, string relativeAssetUrl, out string result)
        {
            result = string.Empty;
            if (!Uri.TryCreate(catalogUrl, UriKind.Absolute, out Uri catalogUri) ||
                string.IsNullOrWhiteSpace(relativeAssetUrl) || relativeAssetUrl.Contains("..") ||
                relativeAssetUrl.StartsWith("/", StringComparison.Ordinal) || Uri.TryCreate(relativeAssetUrl, UriKind.Absolute, out _))
                return false;
            result = new Uri(catalogUri, relativeAssetUrl.Replace('\\', '/')).AbsoluteUri;
            return true;
        }

        private static IEnumerator LoadTile(
            string catalogUrl,
            double[] origin,
            FwAssetReference payloadReference,
            FwAssetReference imageryReference,
            Transform parent,
            float unityUnitsPerMetre,
            string expectedTileId,
            Action<FwLoadedTile> succeeded,
            Action<string> failed)
        {
            if (!TryResolveAssetUrl(catalogUrl, payloadReference.url, out string payloadUrl) ||
                !TryResolveAssetUrl(catalogUrl, imageryReference.url, out string imageryUrl))
            {
                failed?.Invoke("Tile asset URL escapes or does not resolve against the catalog.");
                yield break;
            }
            byte[] payload = null;
            byte[] imagery = null;
            string error = null;
            yield return Download(payloadUrl, payloadReference.sha256, payloadReference.byte_count, (value) => payload = value, (value) => error = value);
            if (error != null)
            {
                failed?.Invoke(error);
                yield break;
            }
            yield return Download(imageryUrl, imageryReference.sha256, imageryReference.byte_count, (value) => imagery = value, (value) => error = value);
            if (error != null)
            {
                failed?.Invoke(error);
                yield break;
            }

            Texture2D texture = null;
            FwTileUnityBuilder builder = null;
            try
            {
                FwDecodedContainer decoded = FwTileContainerDecoder.Decode(payload, new FwUnityJsonParser());
                if (!string.Equals(decoded.Header.tile_id, expectedTileId, StringComparison.Ordinal))
                    throw new FormatException($"Downloaded tile id {decoded.Header.tile_id} differs from {expectedTileId}.");
                FwTileGeometry geometry = FwTileGeometryDecoder.Decode(decoded, origin);
                texture = new Texture2D(2, 2, TextureFormat.RGBA32, true, false)
                {
                    name = $"FireViewer imagery — {expectedTileId}",
                    wrapMode = TextureWrapMode.Clamp,
                    filterMode = FilterMode.Trilinear,
                };
                if (!texture.LoadImage(imagery, false)) throw new InvalidDataException("Unity could not decode the tile imagery.");
                builder = new FwTileUnityBuilder();
                GameObject root = builder.Build(geometry, texture, parent, unityUnitsPerMetre);
                succeeded?.Invoke(new FwLoadedTile(root, texture, geometry, builder));
            }
            catch (Exception exception)
            {
                builder?.Dispose();
                if (texture != null)
                {
                    if (Application.isPlaying) UnityEngine.Object.Destroy(texture);
                    else UnityEngine.Object.DestroyImmediate(texture);
                }
                failed?.Invoke($"Tile {expectedTileId} decode/build failed: {exception.Message}");
            }
        }

        private static IEnumerator Download(
            string url,
            string expectedSha256,
            long expectedBytes,
            Action<byte[]> succeeded,
            Action<string> failed)
        {
            using var request = UnityWebRequest.Get(url);
            request.downloadHandler = new DownloadHandlerBuffer();
            yield return request.SendWebRequest();
            if (request.result != UnityWebRequest.Result.Success)
            {
                failed?.Invoke($"GET {url} failed: {request.responseCode} {request.error}");
                yield break;
            }
            byte[] bytes = request.downloadHandler.data;
            if (expectedBytes > 0 && bytes.LongLength != expectedBytes)
            {
                failed?.Invoke($"GET {url} returned {bytes.LongLength} bytes, expected {expectedBytes}.");
                yield break;
            }
            try
            {
                if (!string.IsNullOrEmpty(expectedSha256)) FwTileContainerDecoder.VerifySha256(bytes, expectedSha256, url);
            }
            catch (Exception exception)
            {
                failed?.Invoke(exception.Message);
                yield break;
            }
            succeeded?.Invoke(bytes);
        }
    }

    [DisallowMultipleComponent]
    public sealed class FwRemoteTileCanary : MonoBehaviour
    {
        [SerializeField] private string catalogUrl = string.Empty;
        [SerializeField] private string tileId = string.Empty;
        [SerializeField] private Transform contentRoot;
        [SerializeField, Min(0.0001f)] private float unityUnitsPerMetre = 1f;
        [SerializeField] private string state = "idle";
        [SerializeField] private string lastError = string.Empty;
        private FwLoadedTile loaded;

        public string State => state;
        public string LastError => lastError;

        public void Configure(string url, string id, Transform parent = null, float scale = 1f)
        {
            catalogUrl = url ?? string.Empty;
            tileId = id ?? string.Empty;
            contentRoot = parent;
            unityUnitsPerMetre = scale;
        }

        public void StartLoad()
        {
            StopAllCoroutines();
            loaded?.Dispose();
            loaded = null;
            state = "loading_catalog";
            lastError = string.Empty;
            StartCoroutine(Load());
        }

        private IEnumerator Load()
        {
            FwRemoteCatalog catalog = null;
            string error = null;
            yield return FwRemoteTileClient.LoadCatalog(catalogUrl, value => catalog = value, value => error = value);
            if (error != null) { Fail(error); yield break; }
            FwCatalogTile selected = null;
            foreach (FwCatalogTile tile in catalog.tiles)
                if (string.Equals(tile.id, tileId, StringComparison.Ordinal)) { selected = tile; break; }
            if (selected == null) { Fail($"Tile {tileId} is absent from the catalog."); yield break; }
            state = "loading_tile";
            yield return FwRemoteTileClient.LoadDetailTile(
                catalogUrl, catalog, selected, contentRoot != null ? contentRoot : transform, unityUnitsPerMetre,
                value => loaded = value, value => error = value);
            if (error != null) { Fail(error); yield break; }
            state = "ready";
        }

        private void Fail(string error)
        {
            state = "error";
            lastError = error;
            Debug.LogError($"FIREVIEWER_REMOTE_TILE_FAILED: {error}", this);
        }

        private void OnDestroy() => loaded?.Dispose();
    }
}
