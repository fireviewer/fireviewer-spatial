using System;
using System.Collections.Generic;
using System.IO;

namespace FireViewer.SpatialTiles
{
    public static class FwTileGeometryDecoder
    {
        private const int TreeRecordBytes = 19;

        public static FwTileGeometry Decode(FwDecodedContainer container, double[] sharedOriginL93)
        {
            if (container == null || sharedOriginL93 == null || sharedOriginL93.Length != 3)
                throw new ArgumentException("Container or shared Lambert-93 origin is invalid.");
            FwContainerHeader header = container.Header;
            var result = new FwTileGeometry { TileId = header.tile_id };
            FwSectionHeader terrain = FindHeader(header, "terrain", required: true);
            result.Terrain = DecodeTerrain(container.Sections["terrain"], terrain.metadata, sharedOriginL93);
            if (container.Sections.TryGetValue("trees", out byte[] trees))
                result.Trees = DecodeTrees(trees, FindHeader(header, "trees", true).metadata, sharedOriginL93);
            result.Buildings = DecodeOptionalMeshes(container, "buildings", sharedOriginL93);
            result.Roads = DecodeOptionalMeshes(container, "roads", sharedOriginL93);
            result.Water = DecodeOptionalMeshes(container, "water", sharedOriginL93);
            return result;
        }

        private static FwMeshData DecodeTerrain(byte[] raw, FwSectionMetadata metadata, double[] origin)
        {
            if (metadata == null || metadata.rows <= 0 || metadata.columns <= 0 || metadata.elevation_quantization == null)
                throw new FormatException("Terrain metadata is incomplete.");
            int sampleCount = checked(metadata.rows * metadata.columns);
            if (raw.Length < sampleCount * 2)
                throw new InvalidDataException("Terrain elevation section is truncated.");
            if (string.Equals(metadata.encoding, "regular-grid-z-u16.v1", StringComparison.Ordinal))
                return DecodeDetailTerrain(raw, metadata, origin, sampleCount);
            if (string.Equals(metadata.encoding, "masked-regular-grid-z-u16.v1", StringComparison.Ordinal))
                return DecodeFarTerrain(raw, metadata, origin, sampleCount);
            throw new FormatException($"Terrain encoding {metadata.encoding} is unsupported.");
        }

        private static FwMeshData DecodeDetailTerrain(byte[] raw, FwSectionMetadata metadata, double[] origin, int sampleCount)
        {
            if (raw.Length != sampleCount * 2 || metadata.sample_spacing_m == null || metadata.sample_spacing_m.Length != 2 ||
                metadata.geometric_bounds_l93_m == null || metadata.geometric_bounds_l93_m.Length != 4)
                throw new InvalidDataException("Detailed terrain dimensions are inconsistent.");
            var vertices = new FwPoint3[sampleCount];
            var uv = new FwPoint2[sampleCount];
            for (int row = 0; row < metadata.rows; row++)
            for (int column = 0; column < metadata.columns; column++)
            {
                int index = row * metadata.columns + column;
                double east = metadata.geometric_bounds_l93_m[0] + column * metadata.sample_spacing_m[0];
                double north = metadata.geometric_bounds_l93_m[3] - row * metadata.sample_spacing_m[1];
                // Detailed packages already store Z as an offset from the
                // shared catalog origin.  Do not subtract origin[2] twice.
                double localUp = DecodeQuantized(FwTileContainerDecoder.ReadUInt16(raw, index * 2), metadata.elevation_quantization);
                vertices[index] = ToUnityLocalUp(east, localUp, north, origin);
                uv[index] = new FwPoint2(column / (float)(metadata.columns - 1), 1f - row / (float)(metadata.rows - 1));
            }
            return new FwMeshData
            {
                Name = "terrain",
                Vertices = vertices,
                Uv = uv,
                Triangles = BuildGridTriangles(metadata.rows, metadata.columns, null),
            };
        }

        private static FwMeshData DecodeFarTerrain(byte[] raw, FwSectionMetadata metadata, double[] origin, int sampleCount)
        {
            if (metadata.sample_spacing_m == null || metadata.sample_spacing_m.Length != 2 || metadata.outer_bounds_l93_m == null ||
                metadata.outer_bounds_l93_m.Length != 4 || metadata.elevation_bytes != sampleCount * 2 ||
                metadata.validity_mask_offset_bytes != metadata.elevation_bytes ||
                raw.Length != metadata.elevation_bytes + metadata.validity_mask_bytes)
                throw new InvalidDataException("Far terrain dimensions or validity mask are inconsistent.");
            var validity = new bool[sampleCount];
            var vertices = new FwPoint3[sampleCount];
            var uv = new FwPoint2[sampleCount];
            int validCount = 0;
            for (int row = 0; row < metadata.rows; row++)
            for (int column = 0; column < metadata.columns; column++)
            {
                int index = row * metadata.columns + column;
                validity[index] = (raw[metadata.validity_mask_offset_bytes + index / 8] & (1 << (index % 8))) != 0;
                if (validity[index]) validCount++;
                double east = metadata.outer_bounds_l93_m[0] + (column + 0.5d) * metadata.sample_spacing_m[0];
                double north = metadata.outer_bounds_l93_m[3] - (row + 0.5d) * metadata.sample_spacing_m[1];
                // Both far and detailed packages store elevation relative to
                // the shared Z origin.  This branch restores the absolute
                // value before the common absolute-to-local conversion.
                double up = origin[2] + DecodeQuantized(FwTileContainerDecoder.ReadUInt16(raw, index * 2), metadata.elevation_quantization);
                vertices[index] = ToUnity(east, up, north, origin);
                uv[index] = new FwPoint2((column + 0.5f) / metadata.columns, 1f - (row + 0.5f) / metadata.rows);
            }
            if (validCount != metadata.valid_sample_count)
                throw new InvalidDataException("Far terrain validity count is inconsistent.");
            return new FwMeshData
            {
                Name = "far-terrain",
                Vertices = vertices,
                Uv = uv,
                Triangles = BuildGridTriangles(metadata.rows, metadata.columns, validity),
            };
        }

        private static int[] BuildGridTriangles(int rows, int columns, bool[] validity)
        {
            var triangles = new List<int>(checked((rows - 1) * (columns - 1) * 6));
            for (int row = 0; row < rows - 1; row++)
            for (int column = 0; column < columns - 1; column++)
            {
                int northwest = row * columns + column;
                int northeast = northwest + 1;
                int southwest = northwest + columns;
                int southeast = southwest + 1;
                AddTriangle(triangles, validity, northwest, northeast, southeast);
                AddTriangle(triangles, validity, northwest, southeast, southwest);
            }
            return triangles.ToArray();
        }

        private static void AddTriangle(List<int> output, bool[] validity, int a, int b, int c)
        {
            if (validity != null && (!validity[a] || !validity[b] || !validity[c])) return;
            output.Add(a);
            output.Add(b);
            output.Add(c);
        }

        private static FwTreeInstance[] DecodeTrees(byte[] raw, FwSectionMetadata metadata, double[] origin)
        {
            bool dimensionsInMillimetres = metadata != null &&
                string.Equals(metadata.encoding, "tree-instance-mm.v1", StringComparison.Ordinal);
            bool dimensionsInCentimetres = metadata != null &&
                string.Equals(metadata.encoding, "tree-instance-position-mm-dimension-cm.v2", StringComparison.Ordinal);
            if ((!dimensionsInMillimetres && !dimensionsInCentimetres) ||
                metadata.record_stride_bytes != TreeRecordBytes || raw.Length != checked(metadata.count * TreeRecordBytes) ||
                metadata.position_origin_l93_m == null || metadata.position_origin_l93_m.Length != 3)
                throw new InvalidDataException("Tree instance section is inconsistent.");
            float dimensionScale = dimensionsInMillimetres ? 1f / 1000f : 1f / 100f;
            var result = new FwTreeInstance[metadata.count];
            for (int index = 0; index < result.Length; index++)
            {
                int offset = index * TreeRecordBytes;
                double east = metadata.position_origin_l93_m[0] + FwTileContainerDecoder.ReadUInt32(raw, offset) / 1000d;
                double north = metadata.position_origin_l93_m[1] + FwTileContainerDecoder.ReadUInt32(raw, offset + 4) / 1000d;
                double localUp = FwTileContainerDecoder.ReadInt32(raw, offset + 8) / 1000d;
                float height = FwTileContainerDecoder.ReadUInt16(raw, offset + 12) * dimensionScale;
                float crown = FwTileContainerDecoder.ReadUInt16(raw, offset + 14) * dimensionScale;
                byte variant = raw[offset + 16];
                float rotation = FwTileContainerDecoder.ReadUInt16(raw, offset + 17) / 100f;
                if (height <= 0f || crown <= 0f)
                    throw new InvalidDataException($"Tree instance {index} has invalid dimensions.");
                result[index] = new FwTreeInstance(ToUnityLocalUp(east, localUp, north, origin), height, crown, variant, rotation);
            }
            return result;
        }

        private static FwMeshData[] DecodeOptionalMeshes(FwDecodedContainer container, string sectionName, double[] origin)
        {
            if (!container.Sections.TryGetValue(sectionName, out byte[] raw)) return Array.Empty<FwMeshData>();
            return DecodeMeshes(raw, FindHeader(container.Header, sectionName, true).metadata, origin);
        }

        private static FwMeshData[] DecodeMeshes(byte[] raw, FwSectionMetadata metadata, double[] origin)
        {
            if (metadata == null || !string.Equals(metadata.encoding, "mesh-position-u16-quantized-index-adaptive.v1", StringComparison.Ordinal) ||
                metadata.vertex_stride_bytes != 6 || metadata.position_quantization == null || metadata.meshes == null ||
                metadata.mesh_count != metadata.meshes.Length)
                throw new InvalidDataException("Vector mesh section metadata is inconsistent.");
            var output = new FwMeshData[metadata.meshes.Length];
            for (int meshIndex = 0; meshIndex < metadata.meshes.Length; meshIndex++)
            {
                FwMeshDescriptor descriptor = metadata.meshes[meshIndex];
                int vertexBytes = checked(descriptor.vertex_count * 6);
                int indexCount = checked(descriptor.triangle_count * 3);
                int indexBytes = checked(indexCount * descriptor.index_stride_bytes);
                if (descriptor.vertex_offset_bytes < 0 || descriptor.index_offset_bytes != descriptor.vertex_offset_bytes + vertexBytes ||
                    descriptor.end_offset_bytes != descriptor.index_offset_bytes + indexBytes || descriptor.end_offset_bytes > raw.Length ||
                    (descriptor.index_stride_bytes != 2 && descriptor.index_stride_bytes != 4))
                    throw new InvalidDataException($"Vector mesh {descriptor.name} byte ranges are inconsistent.");
                var vertices = new FwPoint3[descriptor.vertex_count];
                for (int index = 0; index < vertices.Length; index++)
                {
                    int offset = descriptor.vertex_offset_bytes + index * 6;
                    double east = metadata.position_quantization.east_minimum_m + FwTileContainerDecoder.ReadUInt16(raw, offset) * metadata.position_quantization.east_step_m;
                    double north = metadata.position_quantization.north_minimum_m + FwTileContainerDecoder.ReadUInt16(raw, offset + 2) * metadata.position_quantization.north_step_m;
                    double localUp = metadata.position_quantization.up_minimum_m + FwTileContainerDecoder.ReadUInt16(raw, offset + 4) * metadata.position_quantization.up_step_m;
                    vertices[index] = ToUnityLocalUp(east, localUp, north, origin);
                }
                var indices = new int[indexCount];
                for (int index = 0; index < indices.Length; index++)
                {
                    int offset = descriptor.index_offset_bytes + index * descriptor.index_stride_bytes;
                    indices[index] = descriptor.index_stride_bytes == 2
                        ? FwTileContainerDecoder.ReadUInt16(raw, offset)
                        : checked((int)FwTileContainerDecoder.ReadUInt32(raw, offset));
                    if (indices[index] < 0 || indices[index] >= vertices.Length)
                        throw new InvalidDataException($"Vector mesh {descriptor.name} has an out-of-range index.");
                }
                // (east,north,up) -> Unity (east,up,north) changes handedness.
                // Reverse every source triangle to preserve its visible face.
                for (int triangle = 0; triangle < indices.Length; triangle += 3)
                    (indices[triangle + 1], indices[triangle + 2]) = (indices[triangle + 2], indices[triangle + 1]);
                output[meshIndex] = new FwMeshData { Name = descriptor.name ?? string.Empty, Vertices = vertices, Triangles = indices };
            }
            return output;
        }

        private static FwSectionHeader FindHeader(FwContainerHeader header, string name, bool required)
        {
            foreach (FwSectionHeader section in header.sections)
                if (string.Equals(section.name, name, StringComparison.Ordinal)) return section;
            if (required) throw new FormatException($"FWTile section {name} is missing.");
            return null;
        }

        private static double DecodeQuantized(ushort value, FwQuantization quantization) => quantization.minimum_m + value * quantization.step_m;

        private static FwPoint3 ToUnity(double east, double up, double north, double[] origin) =>
            new FwPoint3(checked((float)(east - origin[0])), checked((float)(up - origin[2])), checked((float)(north - origin[1])));

        private static FwPoint3 ToUnityLocalUp(double east, double localUp, double north, double[] origin) =>
            new FwPoint3(checked((float)(east - origin[0])), checked((float)localUp), checked((float)(north - origin[1])));
    }
}
