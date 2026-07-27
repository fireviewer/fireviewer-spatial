using System;
using System.Collections.Generic;

namespace FireViewer.SpatialTiles
{
    public interface IFwJsonParser
    {
        T Parse<T>(string json);
    }

    [Serializable]
    public sealed class FwQuantization
    {
        public string component_type;
        public double minimum_m;
        public double maximum_m;
        public double step_m;
        public double maximum_observed_error_m;
        public double east_minimum_m;
        public double east_step_m;
        public double north_minimum_m;
        public double north_step_m;
        public double up_minimum_m;
        public double up_step_m;
    }

    [Serializable]
    public sealed class FwMeshDescriptor
    {
        public string name;
        public int vertex_count;
        public int triangle_count;
        public int vertex_offset_bytes;
        public int index_offset_bytes;
        public int end_offset_bytes;
        public string index_component_type;
        public int index_stride_bytes;
    }

    [Serializable]
    public sealed class FwSectionMetadata
    {
        public string encoding;
        public int rows;
        public int columns;
        public double[] sample_spacing_m = Array.Empty<double>();
        public double[] geometric_bounds_l93_m = Array.Empty<double>();
        public double[] outer_bounds_l93_m = Array.Empty<double>();
        public bool sample_centres;
        public int vertex_count;
        public int triangle_count;
        public int elevation_bytes;
        public int validity_mask_offset_bytes;
        public int validity_mask_bytes;
        public int valid_sample_count;
        public FwQuantization elevation_quantization;
        public FwQuantization position_quantization;
        public int record_stride_bytes;
        public int count;
        public double[] position_origin_l93_m = Array.Empty<double>();
        public int vertex_stride_bytes;
        public int mesh_count;
        public FwMeshDescriptor[] meshes = Array.Empty<FwMeshDescriptor>();
    }

    [Serializable]
    public sealed class FwSectionHeader
    {
        public string name;
        public string codec;
        public int offset_bytes;
        public int stored_bytes;
        public int raw_bytes;
        public string stored_sha256;
        public string raw_sha256;
        public FwSectionMetadata metadata;
    }

    [Serializable]
    public sealed class FwContainerHeader
    {
        public string schema;
        public string kind;
        public string tile_id;
        public string crs;
        public string linear_unit;
        public string axis_convention;
        public double[] bounds_l93_m = Array.Empty<double>();
        public double[] origin_l93_m = Array.Empty<double>();
        public FwSectionHeader[] sections = Array.Empty<FwSectionHeader>();
    }

    public sealed class FwDecodedContainer
    {
        public FwDecodedContainer(FwContainerHeader header, Dictionary<string, byte[]> sections)
        {
            Header = header;
            Sections = sections;
        }

        public FwContainerHeader Header { get; }
        public IReadOnlyDictionary<string, byte[]> Sections { get; }
    }

    public readonly struct FwPoint3
    {
        public FwPoint3(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public float X { get; }
        public float Y { get; }
        public float Z { get; }
    }

    public readonly struct FwPoint2
    {
        public FwPoint2(float x, float y)
        {
            X = x;
            Y = y;
        }

        public float X { get; }
        public float Y { get; }
    }

    public sealed class FwMeshData
    {
        public string Name { get; set; } = string.Empty;
        public FwPoint3[] Vertices { get; set; } = Array.Empty<FwPoint3>();
        public FwPoint2[] Uv { get; set; } = Array.Empty<FwPoint2>();
        public int[] Triangles { get; set; } = Array.Empty<int>();
    }

    public readonly struct FwTreeInstance
    {
        public FwTreeInstance(FwPoint3 position, float height, float crownDiameter, byte variant, float rotationDegrees)
        {
            Position = position;
            Height = height;
            CrownDiameter = crownDiameter;
            Variant = variant;
            RotationDegrees = rotationDegrees;
        }

        public FwPoint3 Position { get; }
        public float Height { get; }
        public float CrownDiameter { get; }
        public byte Variant { get; }
        public float RotationDegrees { get; }
    }

    public sealed class FwTileGeometry
    {
        public string TileId { get; set; } = string.Empty;
        public FwMeshData Terrain { get; set; }
        public FwTreeInstance[] Trees { get; set; } = Array.Empty<FwTreeInstance>();
        public FwMeshData[] Buildings { get; set; } = Array.Empty<FwMeshData>();
        public FwMeshData[] Roads { get; set; } = Array.Empty<FwMeshData>();
        public FwMeshData[] Water { get; set; } = Array.Empty<FwMeshData>();
    }

    [Serializable]
    public sealed class FwAssetReference
    {
        public string url;
        public string sha256;
        public long byte_count;
    }

    [Serializable]
    public sealed class FwFarLod
    {
        public string role;
        public FwAssetReference terrain;
        public FwAssetReference imagery;
        public double[] bounds_l93_m = Array.Empty<double>();
    }

    [Serializable]
    public sealed class FwDetailLod
    {
        public bool enabled = true;
        public string mode;
        public double publish_distance_m;
        public double preload_radius_m;
        public int maximum_resident_tile_count;
        public bool near_disabled;
        public string transition;
        public string eviction;
    }

    [Serializable]
    public sealed class FwLodPolicy
    {
        public FwFarLod far;
        public FwDetailLod detail;
    }

    [Serializable]
    public sealed class FwCatalogTile
    {
        public string id;
        public double[] bounds_l93_m = Array.Empty<double>();
        public FwAssetReference payload;
        public FwAssetReference imagery;
        public string[] sections = Array.Empty<string>();

        public double SquaredDistance(double east, double north)
        {
            double dx = east < bounds_l93_m[0] ? bounds_l93_m[0] - east : east > bounds_l93_m[2] ? east - bounds_l93_m[2] : 0d;
            double dz = north < bounds_l93_m[1] ? bounds_l93_m[1] - north : north > bounds_l93_m[3] ? north - bounds_l93_m[3] : 0d;
            return dx * dx + dz * dz;
        }
    }

    [Serializable]
    public sealed class FwRemoteCatalog
    {
        public string schema;
        public int catalog_version;
        public string crs;
        public string linear_unit;
        public double[] origin_l93_m = Array.Empty<double>();
        public FwLodPolicy lod_policy;
        public int exported_detail_tile_count;
        public FwCatalogTile[] tiles = Array.Empty<FwCatalogTile>();

        public void Validate()
        {
            if (!string.Equals(schema, "fireviewer.remote-tile-catalog.v1", StringComparison.Ordinal))
                throw new FormatException("Unsupported FireViewer remote catalog schema.");
            if (!string.Equals(crs, "EPSG:2154", StringComparison.Ordinal) || origin_l93_m == null || origin_l93_m.Length != 3)
                throw new FormatException("Catalog CRS or local origin is invalid.");
            if (lod_policy?.far?.terrain == null || lod_policy.far.imagery == null || lod_policy.detail == null)
                throw new FormatException("Catalog LOD policy is incomplete.");
            if (lod_policy.detail.enabled && (lod_policy.detail.maximum_resident_tile_count <= 0 || lod_policy.detail.maximum_resident_tile_count > 16))
                throw new FormatException("Catalog resident tile budget must be between 1 and 16.");
            if (!lod_policy.detail.enabled &&
                (lod_policy.detail.maximum_resident_tile_count != 0 ||
                 !string.Equals(lod_policy.detail.mode, "global_only", StringComparison.Ordinal) ||
                 !lod_policy.detail.near_disabled))
                throw new FormatException("Global-only catalog declares an invalid detail policy.");
            if (lod_policy.detail.publish_distance_m > lod_policy.detail.preload_radius_m)
                throw new FormatException("Catalog publish distance exceeds preload radius.");
            if (tiles == null || tiles.Length != exported_detail_tile_count)
                throw new FormatException("Catalog exported tile count does not match its tile array.");
            if (!lod_policy.detail.enabled && tiles.Length != 0)
                throw new FormatException("Global-only catalog must not declare detail tiles.");
            var ids = new HashSet<string>(StringComparer.Ordinal);
            foreach (FwCatalogTile tile in tiles)
            {
                if (tile == null || string.IsNullOrWhiteSpace(tile.id) || !ids.Add(tile.id) || tile.bounds_l93_m == null || tile.bounds_l93_m.Length != 4)
                    throw new FormatException("Catalog contains an invalid or duplicate tile.");
                ValidateAsset(tile.payload, $"{tile.id} payload");
                ValidateAsset(tile.imagery, $"{tile.id} imagery");
            }
            ValidateAsset(lod_policy.far.terrain, "far terrain");
            ValidateAsset(lod_policy.far.imagery, "far imagery");
        }

        private static void ValidateAsset(FwAssetReference asset, string label)
        {
            if (asset == null || string.IsNullOrWhiteSpace(asset.url) || asset.byte_count <= 0 || asset.sha256 == null || asset.sha256.Length != 64)
                throw new FormatException($"Catalog {label} reference is invalid.");
            if (Uri.TryCreate(asset.url, UriKind.Absolute, out _) || asset.url.Contains("..") || asset.url.StartsWith("/", StringComparison.Ordinal))
                throw new FormatException($"Catalog {label} URL must be relative and confined.");
        }
    }
}
