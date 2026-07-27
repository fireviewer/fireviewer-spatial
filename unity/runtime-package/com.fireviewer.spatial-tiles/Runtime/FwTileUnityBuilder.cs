using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace FireViewer.SpatialTiles
{
    public sealed class FwTileUnityBuilder : IDisposable
    {
        private const float TreeGroundClearanceMetres = 0.02f;
        private const int FarChunkMaximumVertices = 180_000;
        private const int FarChunkMaximumTriangleIndices = 600_000;
        private readonly Shader surfaceShader;
        private readonly Shader colourShader;
        private readonly Shader buildingShader;
        private readonly Shader treeShader;
        private readonly List<UnityEngine.Object> ownedAssets = new();

        public int LastSourceTreeCount { get; private set; }
        public int LastVisibleTreeCount { get; private set; }
        public int LastMaskedTreeCount { get; private set; }

        public FwTileUnityBuilder()
        {
            surfaceShader = Resources.Load<Shader>("FireViewerOperationalTerrain") ??
                Shader.Find("FireViewer/Operational Terrain") ?? Shader.Find("Unlit/Texture");
            colourShader = Resources.Load<Shader>("FireViewerOperationalFeature") ??
                Shader.Find("FireViewer/Operational Feature") ?? Shader.Find("Standard") ?? Shader.Find("Unlit/Color");
            buildingShader = Resources.Load<Shader>("FireViewerOperationalBuilding") ??
                Shader.Find("FireViewer/Operational Building") ?? colourShader;
            treeShader = Resources.Load<Shader>("FireViewerOperationalVegetation") ??
                Shader.Find("FireViewer/Operational Vegetation") ?? colourShader;
            if (surfaceShader == null || colourShader == null || buildingShader == null || treeShader == null)
                throw new InvalidOperationException("FireViewer operational shaders are unavailable.");
        }

        public GameObject Build(FwTileGeometry geometry, Texture2D orthophoto, Transform parent, float unityUnitsPerMetre = 1f)
        {
            if (geometry?.Terrain == null || orthophoto == null || unityUnitsPerMetre <= 0f)
                throw new ArgumentException("Tile geometry, imagery or Unity scale is invalid.");

            bool isFar = string.Equals(geometry.Terrain.Name, "far-terrain", StringComparison.Ordinal);
            var root = new GameObject($"FireViewer tile — {geometry.TileId}");
            root.transform.SetParent(parent, false);
            root.transform.localScale = Vector3.one * unityUnitsPerMetre;

            Material terrainMaterial = TerrainMaterial(orthophoto, isFar);
            if (isFar)
                BuildChunkedFarTerrain(geometry.Terrain, root.transform, terrainMaterial);
            else
                BuildMeshObject("Terrain", geometry.Terrain, root.transform, terrainMaterial, 0f, ShadowCastingMode.Off, true);

            Material buildings = BuildingMaterial();
            BuildSection("Buildings", geometry.Buildings, root.transform, _ => buildings, ShadowCastingMode.On, true, flatShaded: true);

            Material roadCarriageway = ColourMaterial("Road — carriageway", new Color(0.30f, 0.31f, 0.30f), 2060);
            Material roadShoulder = ColourMaterial("Road — shoulder", new Color(0.38f, 0.36f, 0.32f), 2050);
            Material roadMarking = ColourMaterial("Road — marking", new Color(0.76f, 0.72f, 0.57f), 2070);
            BuildSection(
                "Roads",
                geometry.Roads,
                root.transform,
                mesh => RoadMaterial(mesh.Name, roadCarriageway, roadShoulder, roadMarking),
                ShadowCastingMode.Off,
                false);

            Material waterSegments = ColourMaterial("Water — channel", new Color(0.12f, 0.35f, 0.45f), 2045);
            Material waterSurfaces = ColourMaterial("Water — surface", new Color(0.10f, 0.30f, 0.40f), 2040);
            BuildSection(
                "Water",
                geometry.Water,
                root.transform,
                mesh => mesh.Name.IndexOf("surface", StringComparison.OrdinalIgnoreCase) >= 0 ? waterSurfaces : waterSegments,
                ShadowCastingMode.Off,
                false);

            if (geometry.Trees.Length > 0)
            {
                FwTreeInstance[] visibleTrees = PrepareTrees(geometry, out int maskedTrees, out int groundedTrees);
                LastSourceTreeCount = geometry.Trees.Length;
                LastVisibleTreeCount = visibleTrees.Length;
                LastMaskedTreeCount = maskedTrees;
                if (visibleTrees.Length > 0)
                {
                    var trees = new GameObject("Trees — detected crowns, operational LOD");
                    trees.transform.SetParent(root.transform, false);
                    trees.AddComponent<FwInstancedTreeRenderer>().Configure(visibleTrees, treeShader);
                }
                if (maskedTrees > 0 || groundedTrees > 0)
                    Debug.Log(
                        $"FIREVIEWER_TREE_RENDER_PREPARED tile={geometry.TileId} source={geometry.Trees.Length} " +
                        $"visible={visibleTrees.Length} maskedRoadWaterOrBuilding={maskedTrees} grounded={groundedTrees}");
            }
            return root;
        }

        public void Dispose()
        {
            foreach (UnityEngine.Object asset in ownedAssets)
                if (asset != null) DestroyOwned(asset);
            ownedAssets.Clear();
        }

        private FwTreeInstance[] PrepareTrees(FwTileGeometry geometry, out int maskedTrees, out int groundedTrees)
        {
            var surfaceMask = new FwHorizontalFootprintMask(16f);
            surfaceMask.Add(geometry.Roads);
            surfaceMask.Add(geometry.Water);
            var buildingMask = new FwHorizontalFootprintMask(16f);
            buildingMask.Add(geometry.Buildings);
            var terrain = new FwRegularTerrainSampler(geometry.Terrain);
            var output = new List<FwTreeInstance>(geometry.Trees.Length);
            maskedTrees = 0;
            groundedTrees = 0;

            foreach (FwTreeInstance source in geometry.Trees)
            {
                float crownRadius = Mathf.Max(0.25f, source.CrownDiameter * 0.5f);
                if (surfaceMask.Contains(source.Position.X, source.Position.Z) ||
                    buildingMask.IntersectsCircle(source.Position.X, source.Position.Z, crownRadius))
                {
                    maskedTrees++;
                    continue;
                }

                FwPoint3 position = source.Position;
                if (terrain.TrySample(position.X, position.Z, out float ground))
                {
                    float anchored = ground + TreeGroundClearanceMetres;
                    if (Mathf.Abs(position.Y - anchored) > 0.02f) groundedTrees++;
                    position = new FwPoint3(position.X, anchored, position.Z);
                }
                output.Add(new FwTreeInstance(position, source.Height, source.CrownDiameter, source.Variant, source.RotationDegrees));
            }
            return output.ToArray();
        }

        private void BuildSection(
            string name,
            FwMeshData[] meshes,
            Transform parent,
            Func<FwMeshData, Material> resolveMaterial,
            ShadowCastingMode shadows,
            bool receiveShadows,
            bool flatShaded = false)
        {
            if (meshes == null || meshes.Length == 0) return;
            var section = new GameObject(name);
            section.transform.SetParent(parent, false);
            foreach (FwMeshData mesh in meshes)
                BuildMeshObject(mesh.Name, mesh, section.transform, resolveMaterial(mesh), 0f, shadows, receiveShadows, flatShaded);
        }

        private void BuildChunkedFarTerrain(FwMeshData source, Transform parent, Material material)
        {
            if (source.Vertices == null || source.Vertices.Length == 0 ||
                source.Triangles == null || source.Triangles.Length == 0)
                return;

            var section = new GameObject("Terrain");
            section.transform.SetParent(parent, false);
            var vertexMap = new Dictionary<int, int>(FarChunkMaximumVertices);
            var vertices = new List<Vector3>(FarChunkMaximumVertices);
            var triangles = new List<int>(FarChunkMaximumTriangleIndices);
            bool hasUv = source.Uv != null && source.Uv.Length == source.Vertices.Length;
            var uv = hasUv ? new List<Vector2>(FarChunkMaximumVertices) : null;
            int chunkIndex = 0;

            for (int index = 0; index + 2 < source.Triangles.Length; index += 3)
            {
                int missingVertices = 0;
                for (int corner = 0; corner < 3; corner++)
                    if (!vertexMap.ContainsKey(source.Triangles[index + corner])) missingVertices++;

                if (triangles.Count > 0 &&
                    (vertices.Count + missingVertices > FarChunkMaximumVertices ||
                     triangles.Count + 3 > FarChunkMaximumTriangleIndices))
                {
                    CreateFarChunk(section.transform, material, chunkIndex++, vertices, uv, triangles);
                    vertexMap.Clear();
                    vertices.Clear();
                    triangles.Clear();
                    uv?.Clear();
                }

                for (int corner = 0; corner < 3; corner++)
                {
                    int sourceIndex = source.Triangles[index + corner];
                    if (!vertexMap.TryGetValue(sourceIndex, out int localIndex))
                    {
                        FwPoint3 point = source.Vertices[sourceIndex];
                        localIndex = vertices.Count;
                        vertexMap.Add(sourceIndex, localIndex);
                        vertices.Add(new Vector3(point.X, point.Y, point.Z));
                        if (hasUv) uv.Add(new Vector2(source.Uv[sourceIndex].X, source.Uv[sourceIndex].Y));
                    }
                    triangles.Add(localIndex);
                }
            }

            if (triangles.Count > 0)
                CreateFarChunk(section.transform, material, chunkIndex++, vertices, uv, triangles);

            Debug.Log(
                $"FIREVIEWER_FAR_MESH_CHUNKED chunks={chunkIndex} vertices={source.Vertices.Length} " +
                $"triangles={source.Triangles.Length / 3} maxVerticesPerChunk={FarChunkMaximumVertices}");
        }

        private void CreateFarChunk(
            Transform parent,
            Material material,
            int chunkIndex,
            List<Vector3> vertices,
            List<Vector2> uv,
            List<int> triangles)
        {
            var gameObject = new GameObject($"Terrain {chunkIndex + 1:000}");
            gameObject.transform.SetParent(parent, false);
            var mesh = new Mesh
            {
                name = $"Far terrain chunk {chunkIndex + 1:000}",
                indexFormat = vertices.Count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16,
            };
            mesh.SetVertices(vertices);
            if (uv != null) mesh.SetUVs(0, uv);
            mesh.SetTriangles(triangles, 0, true);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            mesh.UploadMeshData(true);
            gameObject.AddComponent<MeshFilter>().sharedMesh = mesh;
            MeshRenderer renderer = gameObject.AddComponent<MeshRenderer>();
            renderer.sharedMaterial = material;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = true;
            renderer.lightProbeUsage = LightProbeUsage.Off;
            renderer.reflectionProbeUsage = ReflectionProbeUsage.Off;
            ownedAssets.Add(mesh);
        }

        private void BuildMeshObject(
            string name,
            FwMeshData source,
            Transform parent,
            Material material,
            float verticalBiasMetres,
            ShadowCastingMode shadows,
            bool receiveShadows,
            bool flatShaded = false)
        {
            if (source.Vertices == null || source.Vertices.Length == 0 || source.Triangles == null || source.Triangles.Length == 0) return;
            var gameObject = new GameObject(string.IsNullOrWhiteSpace(name) ? "Mesh" : name);
            gameObject.transform.SetParent(parent, false);
            var mesh = new Mesh
            {
                name = $"{gameObject.name} mesh",
                indexFormat = (flatShaded ? source.Triangles.Length : source.Vertices.Length) > 65535
                    ? IndexFormat.UInt32
                    : IndexFormat.UInt16,
            };
            int[] triangles = source.Triangles;
            var vertices = new Vector3[flatShaded ? triangles.Length : source.Vertices.Length];
            if (flatShaded)
            {
                var expandedTriangles = new int[triangles.Length];
                for (int index = 0; index < triangles.Length; index++)
                {
                    FwPoint3 point = source.Vertices[triangles[index]];
                    vertices[index] = new Vector3(point.X, point.Y + verticalBiasMetres, point.Z);
                    expandedTriangles[index] = index;
                }
                triangles = expandedTriangles;
            }
            else
            {
                for (int index = 0; index < vertices.Length; index++)
                    vertices[index] = new Vector3(source.Vertices[index].X, source.Vertices[index].Y + verticalBiasMetres, source.Vertices[index].Z);
            }
            mesh.vertices = vertices;
            mesh.triangles = triangles;
            if (!flatShaded && source.Uv != null && source.Uv.Length == source.Vertices.Length)
            {
                var uv = new Vector2[source.Uv.Length];
                for (int index = 0; index < uv.Length; index++) uv[index] = new Vector2(source.Uv[index].X, source.Uv[index].Y);
                mesh.uv = uv;
            }
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            mesh.UploadMeshData(true);
            gameObject.AddComponent<MeshFilter>().sharedMesh = mesh;
            MeshRenderer renderer = gameObject.AddComponent<MeshRenderer>();
            renderer.sharedMaterial = material;
            renderer.shadowCastingMode = shadows;
            renderer.receiveShadows = receiveShadows;
            renderer.lightProbeUsage = LightProbeUsage.Off;
            renderer.reflectionProbeUsage = ReflectionProbeUsage.Off;
            ownedAssets.Add(mesh);
        }

        private Material TerrainMaterial(Texture texture, bool isFar)
        {
            var material = new Material(surfaceShader) { name = isFar ? "FireViewer terrain FAR" : "FireViewer terrain DETAIL" };
            if (material.HasProperty("_MainTex")) material.SetTexture("_MainTex", texture);
            if (material.HasProperty("_BaseMap")) material.SetTexture("_BaseMap", texture);
            if (material.HasProperty("_Tint")) material.SetColor("_Tint", Color.white);
            if (material.HasProperty("_Color")) material.SetColor("_Color", Color.white);
            if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", Color.white);
            if (material.HasProperty("_ClipDetailFootprints")) material.SetFloat("_ClipDetailFootprints", isFar ? 1f : 0f);
            material.renderQueue = isFar ? 1900 : 1950;
            ownedAssets.Add(material);
            return material;
        }

        private Material ColourMaterial(string name, Color colour, int renderQueue)
        {
            var material = new Material(colourShader) { name = name, enableInstancing = true, renderQueue = renderQueue };
            if (material.HasProperty("_Color")) material.SetColor("_Color", colour);
            if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", colour);
            if (material.HasProperty("_Metallic")) material.SetFloat("_Metallic", 0f);
            if (material.HasProperty("_Glossiness")) material.SetFloat("_Glossiness", 0.04f);
            if (material.HasProperty("_Smoothness")) material.SetFloat("_Smoothness", 0.04f);
            if (material.HasProperty("_SpecColor")) material.SetColor("_SpecColor", Color.black);
            ownedAssets.Add(material);
            return material;
        }

        private Material BuildingMaterial()
        {
            var material = new Material(buildingShader)
            {
                name = "Buildings — operational roof and wall contrast",
                enableInstancing = true,
                renderQueue = 2020,
            };
            if (material.HasProperty("_RoofColor")) material.SetColor("_RoofColor", new Color(0.69f, 0.62f, 0.53f));
            if (material.HasProperty("_WallColor")) material.SetColor("_WallColor", new Color(0.43f, 0.39f, 0.35f));
            if (material.HasProperty("_EdgeColor")) material.SetColor("_EdgeColor", new Color(0.13f, 0.15f, 0.16f));
            if (material.HasProperty("_WindowColor")) material.SetColor("_WindowColor", new Color(0.10f, 0.14f, 0.16f));
            ownedAssets.Add(material);
            return material;
        }

        private static Material RoadMaterial(string name, Material carriageway, Material shoulder, Material marking)
        {
            if (name.IndexOf("mark", StringComparison.OrdinalIgnoreCase) >= 0 ||
                name.IndexOf("center", StringComparison.OrdinalIgnoreCase) >= 0)
                return marking;
            return name.IndexOf("shoulder", StringComparison.OrdinalIgnoreCase) >= 0 ? shoulder : carriageway;
        }

        private static void DestroyOwned(UnityEngine.Object value)
        {
            if (Application.isPlaying) UnityEngine.Object.Destroy(value);
            else UnityEngine.Object.DestroyImmediate(value);
        }
    }

    internal sealed class FwRegularTerrainSampler
    {
        private readonly FwPoint3[] vertices;
        private readonly int rows;
        private readonly int columns;
        private readonly float west;
        private readonly float north;
        private readonly float spacingX;
        private readonly float spacingZ;
        private readonly bool valid;

        public FwRegularTerrainSampler(FwMeshData terrain)
        {
            vertices = terrain?.Vertices ?? Array.Empty<FwPoint3>();
            if (vertices.Length < 4) return;
            north = vertices[0].Z;
            columns = 1;
            while (columns < vertices.Length && Mathf.Abs(vertices[columns].Z - north) < 0.0001f) columns++;
            if (columns < 2 || vertices.Length % columns != 0) return;
            rows = vertices.Length / columns;
            if (rows < 2) return;
            west = vertices[0].X;
            spacingX = vertices[1].X - west;
            spacingZ = north - vertices[columns].Z;
            valid = spacingX > 0f && spacingZ > 0f;
        }

        public bool TrySample(float east, float northing, out float height)
        {
            height = 0f;
            if (!valid) return false;
            float gridX = (east - west) / spacingX;
            float gridZ = (north - northing) / spacingZ;
            if (gridX < -0.001f || gridZ < -0.001f || gridX > columns - 1 + 0.001f || gridZ > rows - 1 + 0.001f)
                return false;
            int column = Mathf.Clamp(Mathf.FloorToInt(gridX), 0, columns - 2);
            int row = Mathf.Clamp(Mathf.FloorToInt(gridZ), 0, rows - 2);
            float x = Mathf.Clamp01(gridX - column);
            float z = Mathf.Clamp01(gridZ - row);
            float nw = vertices[row * columns + column].Y;
            float ne = vertices[row * columns + column + 1].Y;
            float sw = vertices[(row + 1) * columns + column].Y;
            float se = vertices[(row + 1) * columns + column + 1].Y;
            height = x >= z
                ? nw + x * (ne - nw) + z * (se - ne)
                : nw + x * (se - sw) + z * (sw - nw);
            return true;
        }

        public bool TryGetHeightRange(
            float minimumEast,
            float minimumNorthing,
            float maximumEast,
            float maximumNorthing,
            out float minimumHeight,
            out float maximumHeight)
        {
            minimumHeight = 0f;
            maximumHeight = 0f;
            if (!valid) return false;
            if (minimumEast > maximumEast) (minimumEast, maximumEast) = (maximumEast, minimumEast);
            if (minimumNorthing > maximumNorthing) (minimumNorthing, maximumNorthing) = (maximumNorthing, minimumNorthing);

            float eastLimit = west + (columns - 1) * spacingX;
            float southLimit = north - (rows - 1) * spacingZ;
            if (maximumEast < west || minimumEast > eastLimit || maximumNorthing < southLimit || minimumNorthing > north)
                return false;

            int firstColumn = Mathf.Clamp(Mathf.FloorToInt((minimumEast - west) / spacingX), 0, columns - 1);
            int lastColumn = Mathf.Clamp(Mathf.CeilToInt((maximumEast - west) / spacingX), 0, columns - 1);
            int firstRow = Mathf.Clamp(Mathf.FloorToInt((north - maximumNorthing) / spacingZ), 0, rows - 1);
            int lastRow = Mathf.Clamp(Mathf.CeilToInt((north - minimumNorthing) / spacingZ), 0, rows - 1);

            minimumHeight = float.PositiveInfinity;
            maximumHeight = float.NegativeInfinity;
            for (int row = firstRow; row <= lastRow; row++)
            for (int column = firstColumn; column <= lastColumn; column++)
            {
                float height = vertices[row * columns + column].Y;
                if (!float.IsFinite(height)) continue;
                minimumHeight = Mathf.Min(minimumHeight, height);
                maximumHeight = Mathf.Max(maximumHeight, height);
            }
            return float.IsFinite(minimumHeight) && float.IsFinite(maximumHeight);
        }
    }

    internal sealed class FwHorizontalFootprintMask
    {
        private readonly float cellSize;
        private readonly Dictionary<long, List<Triangle>> cells = new();

        public FwHorizontalFootprintMask(float cellSizeMetres)
        {
            cellSize = Mathf.Max(1f, cellSizeMetres);
        }

        public void Add(FwMeshData[] meshes)
        {
            if (meshes == null) return;
            foreach (FwMeshData mesh in meshes)
            {
                if (mesh?.Vertices == null || mesh.Triangles == null) continue;
                for (int index = 0; index + 2 < mesh.Triangles.Length; index += 3)
                {
                    FwPoint3 pa = mesh.Vertices[mesh.Triangles[index]];
                    FwPoint3 pb = mesh.Vertices[mesh.Triangles[index + 1]];
                    FwPoint3 pc = mesh.Vertices[mesh.Triangles[index + 2]];
                    var triangle = new Triangle(pa.X, pa.Z, pb.X, pb.Z, pc.X, pc.Z);
                    if (triangle.IsDegenerate) continue;
                    int minimumX = Cell(triangle.MinimumX);
                    int maximumX = Cell(triangle.MaximumX);
                    int minimumZ = Cell(triangle.MinimumZ);
                    int maximumZ = Cell(triangle.MaximumZ);
                    for (int x = minimumX; x <= maximumX; x++)
                    for (int z = minimumZ; z <= maximumZ; z++)
                    {
                        long key = Key(x, z);
                        if (!cells.TryGetValue(key, out List<Triangle> bucket))
                        {
                            bucket = new List<Triangle>();
                            cells.Add(key, bucket);
                        }
                        bucket.Add(triangle);
                    }
                }
            }
        }

        public bool Contains(float x, float z)
        {
            if (!cells.TryGetValue(Key(Cell(x), Cell(z)), out List<Triangle> bucket)) return false;
            foreach (Triangle triangle in bucket)
                if (triangle.Contains(x, z)) return true;
            return false;
        }

        public bool IntersectsCircle(float x, float z, float radius)
        {
            float safeRadius = Mathf.Max(0f, radius);
            int minimumX = Cell(x - safeRadius);
            int maximumX = Cell(x + safeRadius);
            int minimumZ = Cell(z - safeRadius);
            int maximumZ = Cell(z + safeRadius);
            for (int cellX = minimumX; cellX <= maximumX; cellX++)
            for (int cellZ = minimumZ; cellZ <= maximumZ; cellZ++)
            {
                if (!cells.TryGetValue(Key(cellX, cellZ), out List<Triangle> bucket)) continue;
                foreach (Triangle triangle in bucket)
                    if (triangle.IntersectsCircle(x, z, safeRadius)) return true;
            }
            return false;
        }

        private int Cell(float value) => Mathf.FloorToInt(value / cellSize);
        private static long Key(int x, int z) => ((long)x << 32) ^ (uint)z;

        private readonly struct Triangle
        {
            private readonly float ax;
            private readonly float az;
            private readonly float bx;
            private readonly float bz;
            private readonly float cx;
            private readonly float cz;
            private readonly float denominator;

            public Triangle(float ax, float az, float bx, float bz, float cx, float cz)
            {
                this.ax = ax; this.az = az;
                this.bx = bx; this.bz = bz;
                this.cx = cx; this.cz = cz;
                denominator = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz);
                MinimumX = Mathf.Min(ax, Mathf.Min(bx, cx));
                MaximumX = Mathf.Max(ax, Mathf.Max(bx, cx));
                MinimumZ = Mathf.Min(az, Mathf.Min(bz, cz));
                MaximumZ = Mathf.Max(az, Mathf.Max(bz, cz));
            }

            public float MinimumX { get; }
            public float MaximumX { get; }
            public float MinimumZ { get; }
            public float MaximumZ { get; }
            public bool IsDegenerate => Mathf.Abs(denominator) < 0.000001f;

            public bool Contains(float x, float z)
            {
                float first = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / denominator;
                float second = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) / denominator;
                float third = 1f - first - second;
                const float tolerance = -0.0001f;
                return first >= tolerance && second >= tolerance && third >= tolerance;
            }

            public bool IntersectsCircle(float x, float z, float radius)
            {
                if (Contains(x, z)) return true;
                float squaredRadius = radius * radius;
                if (SquaredDistance(x, z, ax, az) <= squaredRadius ||
                    SquaredDistance(x, z, bx, bz) <= squaredRadius ||
                    SquaredDistance(x, z, cx, cz) <= squaredRadius)
                    return true;
                return SquaredDistanceToSegment(x, z, ax, az, bx, bz) <= squaredRadius ||
                    SquaredDistanceToSegment(x, z, bx, bz, cx, cz) <= squaredRadius ||
                    SquaredDistanceToSegment(x, z, cx, cz, ax, az) <= squaredRadius;
            }

            private static float SquaredDistance(float ax, float az, float bx, float bz)
            {
                float deltaX = ax - bx;
                float deltaZ = az - bz;
                return deltaX * deltaX + deltaZ * deltaZ;
            }

            private static float SquaredDistanceToSegment(
                float pointX,
                float pointZ,
                float startX,
                float startZ,
                float endX,
                float endZ)
            {
                float deltaX = endX - startX;
                float deltaZ = endZ - startZ;
                float squaredLength = deltaX * deltaX + deltaZ * deltaZ;
                if (squaredLength <= 0.0000001f)
                    return SquaredDistance(pointX, pointZ, startX, startZ);
                float factor = Mathf.Clamp01(((pointX - startX) * deltaX + (pointZ - startZ) * deltaZ) / squaredLength);
                return SquaredDistance(pointX, pointZ, startX + factor * deltaX, startZ + factor * deltaZ);
            }
        }
    }

    [DisallowMultipleComponent]
    internal sealed class FwInstancedTreeRenderer : MonoBehaviour
    {
        private const int MaximumBatch = 1023;
        private const int VariantCount = 6;
        private const float NearPrototypeEnterDistanceMetres = 480f;
        private const float NearPrototypeExitDistanceMetres = 560f;
        private const float TrunkDistanceMetres = 440f;
        private const float NearShadowDistanceMetres = 240f;
        private const int DenseForestMidTreeThreshold = 4500;
        private readonly List<TreeBatch> batches = new();
        private readonly Mesh[] midCrowns = new Mesh[VariantCount];
        private readonly Mesh[] nearCrowns = new Mesh[VariantCount];
        private readonly Material[] crownMaterials = new Material[VariantCount];
        private Mesh trunk;
        private Material trunkMaterial;
        private Matrix4x4 cachedLocalToWorld;
        private bool worldMatricesReady;
        private FwSpatialTileStreamingController streaming;

        public void Configure(FwTreeInstance[] trees, Shader shader)
        {
            streaming = GetComponentInParent<FwSpatialTileStreamingController>();
            bool denseForestMid = trees.Length >= DenseForestMidTreeThreshold;
            trunk = BuildTaperedCylinder(8, 0.075f, 0.045f, 0f, 0.56f);
            trunkMaterial = BuildMaterial(shader, "Tree trunk — matte", new Color(0.22f, 0.16f, 0.10f), false);
            Color[] palette =
            {
                new(0.13f, 0.25f, 0.13f), new(0.16f, 0.29f, 0.14f), new(0.19f, 0.31f, 0.15f),
                new(0.09f, 0.20f, 0.13f), new(0.11f, 0.23f, 0.14f), new(0.14f, 0.25f, 0.15f),
            };
            for (int variant = 0; variant < VariantCount; variant++)
            {
                midCrowns[variant] = BuildCrown(variant, false, denseForestMid);
                nearCrowns[variant] = BuildCrown(variant, true, false);
                crownMaterials[variant] = BuildMaterial(shader, $"Tree foliage variant {variant}", palette[variant], true);
            }

            var groups = new List<Matrix4x4>[VariantCount];
            for (int variant = 0; variant < VariantCount; variant++) groups[variant] = new List<Matrix4x4>();
            foreach (FwTreeInstance tree in trees)
            {
                int variant = tree.Variant % VariantCount;
                groups[variant].Add(Matrix4x4.TRS(
                    new Vector3(tree.Position.X, tree.Position.Y, tree.Position.Z),
                    Quaternion.Euler(0f, tree.RotationDegrees, 0f),
                    new Vector3(tree.CrownDiameter, tree.Height, tree.CrownDiameter)));
            }

            for (int variant = 0; variant < VariantCount; variant++)
            for (int start = 0; start < groups[variant].Count; start += MaximumBatch)
            {
                int count = Math.Min(MaximumBatch, groups[variant].Count - start);
                var local = new Matrix4x4[count];
                var world = new Matrix4x4[count];
                Vector3 centre = Vector3.zero;
                for (int offset = 0; offset < count; offset++)
                {
                    local[offset] = groups[variant][start + offset];
                    Vector4 position = local[offset].GetColumn(3);
                    centre += new Vector3(position.x, position.y, position.z);
                }
                batches.Add(new TreeBatch(variant, local, world, centre / count));
            }
        }

        private void LateUpdate()
        {
            Matrix4x4 localToWorld = transform.localToWorldMatrix;
            if (!worldMatricesReady || !MatrixApproximatelyEqual(localToWorld, cachedLocalToWorld))
            {
                foreach (TreeBatch batch in batches)
                    for (int index = 0; index < batch.Local.Length; index++) batch.World[index] = localToWorld * batch.Local[index];
                cachedLocalToWorld = localToWorld;
                worldMatricesReady = true;
            }

            Camera camera = streaming?.ViewerCamera ?? Camera.main;
            float scale = Mathf.Max(0.0001f, Mathf.Abs(transform.lossyScale.x));
            bool nearLodDisabled = streaming?.NearLodDisabled == true;
            bool forceNearBand = !nearLodDisabled && string.Equals(streaming?.Telemetry?.lod_band, "near", StringComparison.Ordinal);
            foreach (TreeBatch batch in batches)
            {
                float distanceMetres = camera == null
                    ? 0f
                    : Vector3.Distance(camera.transform.position, transform.TransformPoint(batch.LocalCentre)) / scale;
                if (nearLodDisabled)
                {
                    batch.UseNear = false;
                }
                else if (forceNearBand)
                {
                    batch.UseNear = true;
                }
                else if (batch.UseNear)
                {
                    if (distanceMetres >= NearPrototypeExitDistanceMetres) batch.UseNear = false;
                }
                else if (distanceMetres <= NearPrototypeEnterDistanceMetres)
                {
                    batch.UseNear = true;
                }
                bool near = batch.UseNear;
                ShadowCastingMode shadows = distanceMetres <= NearShadowDistanceMetres ? ShadowCastingMode.On : ShadowCastingMode.Off;
                if (near && distanceMetres <= TrunkDistanceMetres)
                    Graphics.DrawMeshInstanced(trunk, 0, trunkMaterial, batch.World, batch.World.Length, null, shadows, true, gameObject.layer);
                Graphics.DrawMeshInstanced(
                    near ? nearCrowns[batch.Variant] : midCrowns[batch.Variant],
                    0,
                    crownMaterials[batch.Variant],
                    batch.World,
                    batch.World.Length,
                    null,
                    shadows,
                    true,
                    gameObject.layer);
            }
        }

        private void OnDestroy()
        {
            DestroyRuntime(trunk);
            DestroyRuntime(trunkMaterial);
            for (int variant = 0; variant < VariantCount; variant++)
            {
                DestroyRuntime(midCrowns[variant]);
                DestroyRuntime(nearCrowns[variant]);
                DestroyRuntime(crownMaterials[variant]);
            }
        }

        private static Material BuildMaterial(Shader shader, string name, Color colour, bool foliage)
        {
            var material = new Material(shader) { name = name, enableInstancing = true, renderQueue = 2010 };
            if (material.HasProperty("_Color")) material.SetColor("_Color", colour);
            if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", colour);
            if (material.HasProperty("_Metallic")) material.SetFloat("_Metallic", 0f);
            if (material.HasProperty("_Glossiness")) material.SetFloat("_Glossiness", 0.03f);
            if (material.HasProperty("_Smoothness")) material.SetFloat("_Smoothness", 0.03f);
            if (material.HasProperty("_Foliage")) material.SetFloat("_Foliage", foliage ? 1f : 0f);
            return material;
        }

        private static Mesh BuildCrown(int variant, bool detailed, bool denseForestMid)
        {
            var vertices = new List<Vector3>();
            var triangles = new List<int>();
            if (variant < 3)
            {
                if (!detailed)
                {
                    if (denseForestMid)
                        AppendEllipsoid(vertices, triangles, new Vector3(0f, 0.66f, 0f), new Vector3(0.54f, 0.34f, 0.51f), 8, 4, variant);
                    else
                        AppendEllipsoid(vertices, triangles, new Vector3(0f, 0.72f, 0f), new Vector3(0.46f, 0.31f, 0.42f), 10, 5, variant);
                }
                else
                {
                    AppendOrganicBroadleaf(vertices, triangles, variant);
                }
            }
            else
            {
                if (detailed)
                {
                    AppendLayeredConifer(vertices, triangles, variant);
                }
                else
                {
                    if (denseForestMid)
                    {
                        const int denseSides = 8;
                        AppendCone(vertices, triangles, 0.20f, 0.74f, 0.52f, denseSides, variant);
                        AppendCone(vertices, triangles, 0.48f, 1.00f, 0.38f, denseSides, variant + 3);
                    }
                    else
                    {
                        const int sides = 10;
                        AppendCone(vertices, triangles, 0.28f, 0.72f, 0.48f, sides, variant);
                        AppendCone(vertices, triangles, 0.47f, 0.86f, 0.37f, sides, variant + 2);
                        AppendCone(vertices, triangles, 0.64f, 1.00f, 0.25f, sides, variant + 4);
                    }
                }
            }
            var mesh = new Mesh
            {
                name = detailed
                    ? $"FireViewer near tree crown {variant}"
                    : $"FireViewer mid tree crown {variant}{(denseForestMid ? " dense-forest" : string.Empty)}",
                vertices = vertices.ToArray(),
                triangles = triangles.ToArray(),
            };
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            mesh.UploadMeshData(true);
            return mesh;
        }

        private static void AppendOrganicBroadleaf(List<Vector3> vertices, List<int> triangles, int seed)
        {
            const int sides = 16;
            const int latitudeSegments = 8;
            float centreY = 0.73f + (seed - 1) * 0.012f;
            float halfHeight = 0.31f + (seed % 2) * 0.018f;
            float baseRadius = 0.47f - seed * 0.012f;
            int top = vertices.Count;
            vertices.Add(new Vector3(0.015f * seed, centreY + halfHeight, -0.008f * seed));
            int firstRing = vertices.Count;
            for (int latitude = 1; latitude < latitudeSegments; latitude++)
            {
                float polar = Mathf.PI * latitude / latitudeSegments;
                float profile = Mathf.Pow(Mathf.Sin(polar), 0.68f);
                float ringShiftX = 0.035f * Mathf.Sin(polar * (2.1f + seed * 0.17f));
                float ringShiftZ = 0.030f * Mathf.Cos(polar * (2.6f + seed * 0.11f));
                for (int side = 0; side < sides; side++)
                {
                    float angle = side * Mathf.PI * 2f / sides;
                    float irregular = 1f + 0.075f * Mathf.Sin(angle * (3 + seed) + polar * 2.3f) +
                        0.035f * Mathf.Cos(angle * 5f - seed * 0.8f);
                    vertices.Add(new Vector3(
                        ringShiftX + Mathf.Cos(angle) * baseRadius * profile * irregular,
                        centreY + Mathf.Cos(polar) * halfHeight,
                        ringShiftZ + Mathf.Sin(angle) * baseRadius * profile * (0.92f + 0.025f * seed) * irregular));
                }
            }
            int bottom = vertices.Count;
            vertices.Add(new Vector3(-0.012f * seed, centreY - halfHeight, 0.01f * seed));
            StitchRingSurface(vertices, triangles, top, firstRing, bottom, sides, latitudeSegments - 1);
        }

        private static void AppendLayeredConifer(List<Vector3> vertices, List<int> triangles, int seed)
        {
            const int sides = 12;
            float[] heights = { 0.22f, 0.34f, 0.43f, 0.54f, 0.63f, 0.74f, 0.84f, 0.93f };
            float[] radii = { 0.39f, 0.49f, 0.28f, 0.41f, 0.22f, 0.32f, 0.14f, 0.20f };
            int firstRing = vertices.Count;
            for (int ring = 0; ring < heights.Length; ring++)
            {
                float shiftX = 0.018f * Mathf.Sin(seed * 0.8f + ring * 1.7f);
                float shiftZ = 0.018f * Mathf.Cos(seed * 0.6f + ring * 1.4f);
                for (int side = 0; side < sides; side++)
                {
                    float angle = side * Mathf.PI * 2f / sides;
                    float irregular = 1f + 0.055f * Mathf.Sin(angle * (3 + seed % 3) + ring * 0.63f);
                    vertices.Add(new Vector3(
                        shiftX + Mathf.Cos(angle) * radii[ring] * irregular,
                        heights[ring],
                        shiftZ + Mathf.Sin(angle) * radii[ring] * irregular));
                }
            }
            int tip = vertices.Count;
            vertices.Add(new Vector3(0.008f * Mathf.Sin(seed), 1.03f, 0.008f * Mathf.Cos(seed)));
            int baseCentre = vertices.Count;
            vertices.Add(new Vector3(0f, heights[0], 0f));
            for (int ring = 0; ring < heights.Length - 1; ring++)
            {
                int current = firstRing + ring * sides;
                int nextRing = current + sides;
                for (int side = 0; side < sides; side++)
                {
                    int next = (side + 1) % sides;
                    triangles.Add(current + side); triangles.Add(nextRing + side); triangles.Add(nextRing + next);
                    triangles.Add(current + side); triangles.Add(nextRing + next); triangles.Add(current + next);
                }
            }
            int lastRing = firstRing + (heights.Length - 1) * sides;
            for (int side = 0; side < sides; side++)
            {
                int next = (side + 1) % sides;
                triangles.Add(lastRing + side); triangles.Add(tip); triangles.Add(lastRing + next);
                triangles.Add(baseCentre); triangles.Add(firstRing + next); triangles.Add(firstRing + side);
            }
        }

        private static void StitchRingSurface(
            List<Vector3> vertices,
            List<int> triangles,
            int top,
            int firstRing,
            int bottom,
            int sides,
            int ringCount)
        {
            for (int side = 0; side < sides; side++)
            {
                int next = (side + 1) % sides;
                triangles.Add(top); triangles.Add(firstRing + side); triangles.Add(firstRing + next);
            }
            for (int ring = 0; ring < ringCount - 1; ring++)
            {
                int current = firstRing + ring * sides;
                int nextRing = current + sides;
                for (int side = 0; side < sides; side++)
                {
                    int next = (side + 1) % sides;
                    triangles.Add(current + side); triangles.Add(nextRing + side); triangles.Add(nextRing + next);
                    triangles.Add(current + side); triangles.Add(nextRing + next); triangles.Add(current + next);
                }
            }
            int lastRing = firstRing + (ringCount - 1) * sides;
            for (int side = 0; side < sides; side++)
            {
                int next = (side + 1) % sides;
                triangles.Add(lastRing + side); triangles.Add(bottom); triangles.Add(lastRing + next);
            }
        }

        private static void AppendEllipsoid(
            List<Vector3> vertices,
            List<int> triangles,
            Vector3 centre,
            Vector3 radius,
            int sides,
            int latitudeSegments,
            int seed)
        {
            int top = vertices.Count;
            vertices.Add(centre + Vector3.up * radius.y);
            int firstRing = vertices.Count;
            for (int latitude = 1; latitude < latitudeSegments; latitude++)
            {
                float polar = Mathf.PI * latitude / latitudeSegments;
                float ring = Mathf.Sin(polar);
                for (int side = 0; side < sides; side++)
                {
                    float angle = side * Mathf.PI * 2f / sides;
                    float irregular = 1f + 0.055f * Mathf.Sin(angle * (3 + seed % 3) + seed * 0.73f);
                    vertices.Add(centre + new Vector3(
                        Mathf.Cos(angle) * radius.x * ring * irregular,
                        Mathf.Cos(polar) * radius.y,
                        Mathf.Sin(angle) * radius.z * ring * irregular));
                }
            }
            int bottom = vertices.Count;
            vertices.Add(centre - Vector3.up * radius.y);
            for (int side = 0; side < sides; side++)
            {
                int next = (side + 1) % sides;
                triangles.Add(top); triangles.Add(firstRing + side); triangles.Add(firstRing + next);
            }
            for (int latitude = 0; latitude < latitudeSegments - 2; latitude++)
            {
                int first = firstRing + latitude * sides;
                int second = first + sides;
                for (int side = 0; side < sides; side++)
                {
                    int next = (side + 1) % sides;
                    triangles.Add(first + side); triangles.Add(second + side); triangles.Add(second + next);
                    triangles.Add(first + side); triangles.Add(second + next); triangles.Add(first + next);
                }
            }
            int lastRing = firstRing + (latitudeSegments - 2) * sides;
            for (int side = 0; side < sides; side++)
            {
                int next = (side + 1) % sides;
                triangles.Add(lastRing + side); triangles.Add(bottom); triangles.Add(lastRing + next);
            }
        }

        private static void AppendCone(
            List<Vector3> vertices,
            List<int> triangles,
            float baseY,
            float tipY,
            float radius,
            int sides,
            int seed)
        {
            int baseStart = vertices.Count;
            for (int side = 0; side < sides; side++)
            {
                float angle = side * Mathf.PI * 2f / sides;
                float irregular = 1f + 0.045f * Mathf.Sin(angle * (3 + seed % 4) + seed * 0.51f);
                vertices.Add(new Vector3(Mathf.Cos(angle) * radius * irregular, baseY, Mathf.Sin(angle) * radius * irregular));
            }
            int tip = vertices.Count;
            vertices.Add(new Vector3(0f, tipY, 0f));
            int baseCentre = vertices.Count;
            vertices.Add(new Vector3(0f, baseY, 0f));
            for (int side = 0; side < sides; side++)
            {
                int next = (side + 1) % sides;
                triangles.Add(baseStart + side); triangles.Add(tip); triangles.Add(baseStart + next);
                triangles.Add(baseStart + side); triangles.Add(baseStart + next); triangles.Add(baseCentre);
            }
        }

        private static Mesh BuildTaperedCylinder(int sides, float bottomRadius, float topRadius, float bottomY, float topY)
        {
            var vertices = new Vector3[sides * 2 + 2];
            var triangles = new int[sides * 12];
            int bottomCentre = sides * 2;
            int topCentre = bottomCentre + 1;
            vertices[bottomCentre] = new Vector3(0f, bottomY, 0f);
            vertices[topCentre] = new Vector3(0f, topY, 0f);
            for (int index = 0; index < sides; index++)
            {
                float angle = index * Mathf.PI * 2f / sides;
                vertices[index] = new Vector3(Mathf.Cos(angle) * bottomRadius, bottomY, Mathf.Sin(angle) * bottomRadius);
                vertices[index + sides] = new Vector3(Mathf.Cos(angle) * topRadius, topY, Mathf.Sin(angle) * topRadius);
                int next = (index + 1) % sides;
                int offset = index * 12;
                triangles[offset] = index; triangles[offset + 1] = index + sides; triangles[offset + 2] = next + sides;
                triangles[offset + 3] = index; triangles[offset + 4] = next + sides; triangles[offset + 5] = next;
                triangles[offset + 6] = bottomCentre; triangles[offset + 7] = next; triangles[offset + 8] = index;
                triangles[offset + 9] = topCentre; triangles[offset + 10] = index + sides; triangles[offset + 11] = next + sides;
            }
            var mesh = new Mesh { name = "FireViewer instanced tapered trunk", vertices = vertices, triangles = triangles };
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            mesh.UploadMeshData(true);
            return mesh;
        }

        private static bool MatrixApproximatelyEqual(Matrix4x4 first, Matrix4x4 second)
        {
            for (int row = 0; row < 4; row++)
            for (int column = 0; column < 4; column++)
                if (Mathf.Abs(first[row, column] - second[row, column]) > 0.000001f) return false;
            return true;
        }

        private static void DestroyRuntime(UnityEngine.Object value)
        {
            if (value == null) return;
            if (Application.isPlaying) Destroy(value);
            else DestroyImmediate(value);
        }

        private sealed class TreeBatch
        {
            public TreeBatch(int variant, Matrix4x4[] local, Matrix4x4[] world, Vector3 localCentre)
            {
                Variant = variant;
                Local = local;
                World = world;
                LocalCentre = localCentre;
            }

            public int Variant { get; }
            public Matrix4x4[] Local { get; }
            public Matrix4x4[] World { get; }
            public Vector3 LocalCentre { get; }
            public bool UseNear { get; set; }
        }
    }
}
