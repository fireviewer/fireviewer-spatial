using System.Security.Cryptography;
using System.Text.Json;
using FireViewer.SpatialTiles;

if (args.Length != 1 || !Uri.TryCreate(args[0], UriKind.Absolute, out Uri catalogUri))
    throw new ArgumentException("Usage: FireViewer.TileProbe <absolute-catalog-url>");

using var http = new HttpClient();
string catalogJson = await http.GetStringAsync(catalogUri);
var parser = new SystemTextJsonParser();
FwRemoteCatalog catalog = parser.Parse<FwRemoteCatalog>(catalogJson);
catalog.Validate();
if (catalog.tiles.Length == 0) throw new InvalidDataException("Catalog has no detail tile.");
FwCatalogTile tile = catalog.tiles[0];
var payloadUri = new Uri(catalogUri, tile.payload.url);
byte[] payload = await http.GetByteArrayAsync(payloadUri);
if (payload.LongLength != tile.payload.byte_count) throw new InvalidDataException("Remote payload byte count mismatch.");
FwTileContainerDecoder.VerifySha256(payload, tile.payload.sha256, "remote payload");
FwDecodedContainer container = FwTileContainerDecoder.Decode(payload, parser);
FwTileGeometry geometry = FwTileGeometryDecoder.Decode(container, catalog.origin_l93_m);
if (geometry.Terrain.Vertices.Length == 0 || geometry.Terrain.Triangles.Length == 0)
    throw new InvalidDataException("Remote terrain did not become a mesh.");
if (geometry.Trees.Length == 0)
    throw new InvalidDataException("Remote tree records were not decoded.");
FwAssetReference farReference = catalog.lod_policy.far.terrain;
var farUri = new Uri(catalogUri, farReference.url);
byte[] farPayload = await http.GetByteArrayAsync(farUri);
if (farPayload.LongLength != farReference.byte_count) throw new InvalidDataException("Remote far payload byte count mismatch.");
FwTileContainerDecoder.VerifySha256(farPayload, farReference.sha256, "remote far payload");
FwTileGeometry farGeometry = FwTileGeometryDecoder.Decode(FwTileContainerDecoder.Decode(farPayload, parser), catalog.origin_l93_m);
FwTileSelectionPlan nearPlan = FwSpatialLodPlanner.Select(catalog, 1d, 1d, 750d, 16);
FwTileSelectionPlan midPlan = FwSpatialLodPlanner.Select(catalog, 1d, 1d, 3000d, 16);
FwTileSelectionPlan farPlan = FwSpatialLodPlanner.Select(catalog, 1d, 1d, 3000.001d, 16);
var unorderedFootprint = new FwPlanarFootprint(new[]
{
    new FwPlanarPoint(2d, 2d), new FwPlanarPoint(-1d, -1d),
    new FwPlanarPoint(2d, -1d), new FwPlanarPoint(-1d, 2d),
    new FwPlanarPoint(2d, 2d),
}, 0d);
bool unorderedFootprintIntersects = unorderedFootprint.Intersects(tile);
var exactMarginPolygon = new[]
{
    new FwPlanarPoint(52d, 0.5d), new FwPlanarPoint(53d, 0.5d),
    new FwPlanarPoint(53d, 1.5d), new FwPlanarPoint(52d, 1.5d),
};
bool margin50Includes = new FwPlanarFootprint(exactMarginPolygon, 50d).Intersects(tile);
bool margin49Point9Excludes = !new FwPlanarFootprint(exactMarginPolygon, 49.9d).Intersects(tile);
var overloadedTiles = new FwCatalogTile[17];
for (int index = 0; index < overloadedTiles.Length; index++)
    overloadedTiles[index] = new FwCatalogTile { id = $"tile-{index:00}", bounds_l93_m = new[] { 0d, 0d, 2d, 2d } };
FwCatalogTile[] originalTiles = catalog.tiles;
catalog.tiles = overloadedTiles;
FwTileSelectionPlan clampedPlan = FwSpatialLodPlanner.Select(catalog, 1d, 1d, 1000d, 16);
catalog.tiles = new[] { new FwCatalogTile { id = "visible-beyond-focus-radius", bounds_l93_m = new[] { 800d, 0d, 900d, 100d } } };
var remoteFootprint = new FwPlanarFootprint(new[]
{
    new FwPlanarPoint(790d, -10d), new FwPlanarPoint(910d, -10d),
    new FwPlanarPoint(910d, 110d), new FwPlanarPoint(790d, 110d),
}, 0d);
FwTileSelectionPlan visibleBeyondRadiusPlan = FwSpatialLodPlanner.Select(catalog, 0d, 0d, 120d, 16, remoteFootprint);
FwTileSelectionPlan midVisibleFootprintPlan = FwSpatialLodPlanner.Select(catalog, 0d, 0d, 1400d, 16, remoteFootprint);
catalog.tiles = originalTiles;
var atomic = new FwAtomicPublicationState();
atomic.Begin(new[] { "a", "b" });
atomic.Stage("a");
bool partialPublished = atomic.TryPublish();
bool farDuringPartial = atomic.FarVisible;
atomic.Stage("b");
bool completePublished = atomic.TryPublish();
bool farAfterPublication = atomic.FarVisible;
atomic.Fail("synthetic failure");
bool farAfterFailure = atomic.FarVisible && !atomic.DetailVisible;

Console.WriteLine(JsonSerializer.Serialize(new
{
    schema = catalog.schema,
    tile_id = geometry.TileId,
    payload_url = payloadUri.AbsoluteUri,
    payload_sha256 = Convert.ToHexString(SHA256.HashData(payload)).ToLowerInvariant(),
    terrain_vertices = geometry.Terrain.Vertices.Length,
    terrain_triangles = geometry.Terrain.Triangles.Length / 3,
    terrain_first_y = geometry.Terrain.Vertices[0].Y,
    trees = geometry.Trees.Length,
    tree_first_y = geometry.Trees[0].Position.Y,
    building_meshes = geometry.Buildings.Length,
    road_meshes = geometry.Roads.Length,
    water_meshes = geometry.Water.Length,
    far_terrain_vertices = farGeometry.Terrain.Vertices.Length,
    far_terrain_triangles = farGeometry.Terrain.Triangles.Length / 3,
    far_terrain_first_y = farGeometry.Terrain.Vertices[0].Y,
    near_750_tiles = nearPlan.Tiles.Length,
    mid_3000_tiles = midPlan.Tiles.Length,
    far_over_3000_tiles = farPlan.Tiles.Length,
    unordered_footprint_intersects = unorderedFootprintIntersects,
    margin_50_includes = margin50Includes,
    margin_49_9_excludes = margin49Point9Excludes,
    visible_beyond_radius_tiles = visibleBeyondRadiusPlan.Tiles.Length,
    mid_visible_footprint_tiles = midVisibleFootprintPlan.Tiles.Length,
    band_750 = FwSpatialLodPlanner.ClassifyBand(750d),
    band_over_750 = FwSpatialLodPlanner.ClassifyBand(750.001d),
    band_3000 = FwSpatialLodPlanner.ClassifyBand(3000d),
    band_over_3000 = FwSpatialLodPlanner.ClassifyBand(3000.001d),
    catalog_near_disabled = catalog.lod_policy.detail.near_disabled,
    band_300_near_disabled = FwSpatialLodPlanner.ClassifyBand(300d, catalog.lod_policy.detail.near_disabled),
    budget_clamped_tiles = clampedPlan.Tiles.Length,
    budget_clamped_without_blocking = !clampedPlan.IsBlocked,
    partial_published = partialPublished,
    complete_published = completePublished,
    far_during_partial = farDuringPartial,
    far_after_publication = farAfterPublication,
    far_after_failure = farAfterFailure,
}));

sealed class SystemTextJsonParser : IFwJsonParser
{
    private static readonly JsonSerializerOptions Options = new()
    {
        IncludeFields = true,
        PropertyNameCaseInsensitive = false,
    };

    public T Parse<T>(string json) => JsonSerializer.Deserialize<T>(json, Options)
        ?? throw new FormatException($"JSON produced no {typeof(T).Name} object.");
}
