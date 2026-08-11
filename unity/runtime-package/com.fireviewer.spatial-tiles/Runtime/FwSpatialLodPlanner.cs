using System;
using System.Collections.Generic;

namespace FireViewer.SpatialTiles
{
    public readonly struct FwFocusZone
    {
        public FwFocusZone(string id, string label, double easting, double northing, double elevation)
        {
            Id = id;
            Label = label;
            Easting = easting;
            Northing = northing;
            Elevation = elevation;
        }

        public string Id { get; }
        public string Label { get; }
        public double Easting { get; }
        public double Northing { get; }
        public double Elevation { get; }
    }

    public static class FwKnownFocusZones
    {
        // Elevations are sampled from the catalog's 5 m global MNT at each
        // focus coordinate.  A correct vertical target is essential in the
        // near band: looking at the local-frame zero would aim below ground.
        public static readonly FwFocusZone Montmaur = new("montmaur", "Montmaur-en-Diois", 888576.5d, 6400287.5d, 520.897d);
        public static readonly FwFocusZone Barsac = new("barsac", "Barsac", 881380.5d, 6406209.5d, 386.399d);
        public static readonly FwFocusZone Ausson = new("ausson", "Ausson", 889056d, 6405524d, 448.392d);

        public static bool TryGet(string id, out FwFocusZone zone)
        {
            if (string.Equals(id, Montmaur.Id, StringComparison.OrdinalIgnoreCase)) { zone = Montmaur; return true; }
            if (string.Equals(id, Barsac.Id, StringComparison.OrdinalIgnoreCase)) { zone = Barsac; return true; }
            if (string.Equals(id, Ausson.Id, StringComparison.OrdinalIgnoreCase)) { zone = Ausson; return true; }
            zone = default;
            return false;
        }
    }

    public sealed class FwTileSelectionPlan
    {
        internal FwTileSelectionPlan(FwCatalogTile[] tiles, string blockingError)
        {
            Tiles = tiles;
            BlockingError = blockingError ?? string.Empty;
        }

        public FwCatalogTile[] Tiles { get; }
        public string BlockingError { get; }
        public bool IsBlocked => BlockingError.Length > 0;
    }

    public readonly struct FwPlanarPoint
    {
        public FwPlanarPoint(double east, double north)
        {
            East = east;
            North = north;
        }

        public double East { get; }
        public double North { get; }
    }

    /// <summary>
    /// Convex camera footprint expressed in Lambert-93 metres.  Tile bounds
    /// are expanded by MarginMetres so a small camera movement does not cause
    /// a visible load/unload flicker at the edge of the viewport.
    /// </summary>
    public sealed class FwPlanarFootprint
    {
        private readonly FwPlanarPoint[] points;

        public FwPlanarFootprint(FwPlanarPoint[] vertices, double marginMetres)
        {
            if (vertices == null || vertices.Length < 3) throw new ArgumentException("A planar footprint requires at least three points.");
            if (!IsFinite(marginMetres) || marginMetres < 0d) throw new ArgumentException("Footprint margin is invalid.");
            foreach (FwPlanarPoint point in vertices)
                if (!IsFinite(point.East) || !IsFinite(point.North)) throw new ArgumentException("Footprint point is not finite.");
            points = ConvexHull(vertices);
            if (points.Length < 3) throw new ArgumentException("Planar footprint area is degenerate.");
            MarginMetres = marginMetres;
        }

        public double MarginMetres { get; }

        public bool Intersects(FwCatalogTile tile)
        {
            if (tile?.bounds_l93_m == null || tile.bounds_l93_m.Length != 4) return false;
            double minimumEast = tile.bounds_l93_m[0] - MarginMetres;
            double minimumNorth = tile.bounds_l93_m[1] - MarginMetres;
            double maximumEast = tile.bounds_l93_m[2] + MarginMetres;
            double maximumNorth = tile.bounds_l93_m[3] + MarginMetres;

            foreach (FwPlanarPoint point in points)
                if (InsideRectangle(point, minimumEast, minimumNorth, maximumEast, maximumNorth)) return true;

            var corners = new[]
            {
                new FwPlanarPoint(minimumEast, minimumNorth),
                new FwPlanarPoint(maximumEast, minimumNorth),
                new FwPlanarPoint(maximumEast, maximumNorth),
                new FwPlanarPoint(minimumEast, maximumNorth),
            };
            foreach (FwPlanarPoint corner in corners)
                if (InsidePolygon(corner)) return true;

            for (int pointIndex = 0; pointIndex < points.Length; pointIndex++)
            {
                FwPlanarPoint start = points[pointIndex];
                FwPlanarPoint end = points[(pointIndex + 1) % points.Length];
                for (int edgeIndex = 0; edgeIndex < corners.Length; edgeIndex++)
                    if (SegmentsIntersect(start, end, corners[edgeIndex], corners[(edgeIndex + 1) % corners.Length])) return true;
            }
            return false;
        }

        private bool InsidePolygon(FwPlanarPoint point)
        {
            bool inside = false;
            for (int current = 0, previous = points.Length - 1; current < points.Length; previous = current++)
            {
                FwPlanarPoint a = points[current];
                FwPlanarPoint b = points[previous];
                bool crosses = (a.North > point.North) != (b.North > point.North) &&
                    point.East < (b.East - a.East) * (point.North - a.North) / (b.North - a.North) + a.East;
                if (crosses) inside = !inside;
            }
            return inside;
        }

        private static bool InsideRectangle(FwPlanarPoint point, double minimumEast, double minimumNorth, double maximumEast, double maximumNorth)
            => point.East >= minimumEast && point.East <= maximumEast && point.North >= minimumNorth && point.North <= maximumNorth;

        private static bool SegmentsIntersect(FwPlanarPoint a, FwPlanarPoint b, FwPlanarPoint c, FwPlanarPoint d)
        {
            double abC = Cross(a, b, c);
            double abD = Cross(a, b, d);
            double cdA = Cross(c, d, a);
            double cdB = Cross(c, d, b);
            const double epsilon = 0.000001d;
            if (Math.Abs(abC) <= epsilon && OnSegment(a, b, c)) return true;
            if (Math.Abs(abD) <= epsilon && OnSegment(a, b, d)) return true;
            if (Math.Abs(cdA) <= epsilon && OnSegment(c, d, a)) return true;
            if (Math.Abs(cdB) <= epsilon && OnSegment(c, d, b)) return true;
            return (abC > 0d) != (abD > 0d) && (cdA > 0d) != (cdB > 0d);
        }

        private static double Cross(FwPlanarPoint a, FwPlanarPoint b, FwPlanarPoint c)
            => (b.East - a.East) * (c.North - a.North) - (b.North - a.North) * (c.East - a.East);

        private static bool OnSegment(FwPlanarPoint a, FwPlanarPoint b, FwPlanarPoint point)
            => point.East >= Math.Min(a.East, b.East) && point.East <= Math.Max(a.East, b.East) &&
               point.North >= Math.Min(a.North, b.North) && point.North <= Math.Max(a.North, b.North);

        private static FwPlanarPoint[] ConvexHull(FwPlanarPoint[] input)
        {
            var ordered = new List<FwPlanarPoint>(input);
            ordered.Sort((left, right) =>
            {
                int east = left.East.CompareTo(right.East);
                return east != 0 ? east : left.North.CompareTo(right.North);
            });
            for (int index = ordered.Count - 1; index > 0; index--)
                if (Math.Abs(ordered[index].East - ordered[index - 1].East) <= 0.000001d &&
                    Math.Abs(ordered[index].North - ordered[index - 1].North) <= 0.000001d)
                    ordered.RemoveAt(index);
            if (ordered.Count < 3) return Array.Empty<FwPlanarPoint>();

            var hull = new List<FwPlanarPoint>(ordered.Count * 2);
            foreach (FwPlanarPoint point in ordered)
            {
                while (hull.Count >= 2 && Cross(hull[hull.Count - 2], hull[hull.Count - 1], point) <= 0d) hull.RemoveAt(hull.Count - 1);
                hull.Add(point);
            }
            int lowerCount = hull.Count;
            for (int index = ordered.Count - 2; index >= 0; index--)
            {
                FwPlanarPoint point = ordered[index];
                while (hull.Count > lowerCount && Cross(hull[hull.Count - 2], hull[hull.Count - 1], point) <= 0d) hull.RemoveAt(hull.Count - 1);
                hull.Add(point);
            }
            if (hull.Count > 1) hull.RemoveAt(hull.Count - 1);
            return hull.ToArray();
        }

        private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }

    public static class FwSpatialLodPlanner
    {
        public const int AbsoluteMaximumResidentTiles = 16;
        public const double NearMaximumMetres = 750d;
        public const double MidMaximumMetres = 3000d;

        public static string ClassifyBand(double viewDistanceMetres)
            => ClassifyBand(viewDistanceMetres, false);

        public static string ClassifyBand(double viewDistanceMetres, bool nearDisabled)
        {
            if (!IsFinite(viewDistanceMetres) || viewDistanceMetres < 0d) throw new ArgumentException("View distance is invalid.");
            if (!nearDisabled && viewDistanceMetres <= NearMaximumMetres) return "near";
            if (viewDistanceMetres <= MidMaximumMetres) return "mid";
            return "far";
        }

        public static FwTileSelectionPlan Select(FwRemoteCatalog catalog, double easting, double northing, int runtimeBudget)
            => Select(catalog, easting, northing, 0d, runtimeBudget);

        public static FwTileSelectionPlan Select(FwRemoteCatalog catalog, double easting, double northing, double viewDistanceMetres, int runtimeBudget)
            => Select(catalog, easting, northing, viewDistanceMetres, runtimeBudget, null);

        public static FwTileSelectionPlan Select(
            FwRemoteCatalog catalog,
            double easting,
            double northing,
            double viewDistanceMetres,
            int runtimeBudget,
            FwPlanarFootprint visibleFootprint)
        {
            if (catalog?.lod_policy?.detail == null) throw new ArgumentException("Catalog detail LOD is missing.");
            if (!IsFinite(easting) || !IsFinite(northing)) throw new ArgumentException("Lambert-93 focus is not finite.");
            int budget = Math.Min(AbsoluteMaximumResidentTiles, Math.Min(runtimeBudget, catalog.lod_policy.detail.maximum_resident_tile_count));
            if (budget <= 0) return new FwTileSelectionPlan(Array.Empty<FwCatalogTile>(), "Resident tile budget is zero.");
            string band = ClassifyBand(viewDistanceMetres, catalog.lod_policy.detail.near_disabled);
            if (string.Equals(band, "far", StringComparison.Ordinal))
                return new FwTileSelectionPlan(Array.Empty<FwCatalogTile>(), string.Empty);
            double radiusSquared = catalog.lod_policy.detail.preload_radius_m * catalog.lod_policy.detail.preload_radius_m;
            var candidates = new List<Candidate>();
            foreach (FwCatalogTile tile in catalog.tiles)
            {
                double distance = tile.SquaredDistance(easting, northing);
                bool includeTile = visibleFootprint != null
                    ? visibleFootprint.Intersects(tile)
                    : distance <= radiusSquared;
                if (includeTile) candidates.Add(new Candidate(tile, distance));
            }
            candidates.Sort(Candidate.Compare);
            int selectedCount = Math.Min(candidates.Count, budget);
            var selected = new FwCatalogTile[selectedCount];
            for (int index = 0; index < selected.Length; index++) selected[index] = candidates[index].Tile;
            return new FwTileSelectionPlan(selected, string.Empty);
        }

        private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

        private readonly struct Candidate
        {
            public Candidate(FwCatalogTile tile, double distance) { Tile = tile; Distance = distance; }
            public FwCatalogTile Tile { get; }
            private double Distance { get; }
            public static int Compare(Candidate left, Candidate right)
            {
                int byDistance = left.Distance.CompareTo(right.Distance);
                return byDistance != 0 ? byDistance : string.Compare(left.Tile.id, right.Tile.id, StringComparison.Ordinal);
            }
        }
    }

    public sealed class FwAtomicPublicationState
    {
        private readonly HashSet<string> desired = new(StringComparer.Ordinal);
        private readonly HashSet<string> staged = new(StringComparer.Ordinal);

        public bool FarVisible { get; private set; } = true;
        public bool DetailVisible { get; private set; }
        public string Failure { get; private set; } = string.Empty;

        public void Begin(IEnumerable<string> tileIds)
        {
            desired.Clear(); staged.Clear(); Failure = string.Empty;
            foreach (string id in tileIds) desired.Add(id);
            FarVisible = true; DetailVisible = false;
        }

        public bool Stage(string tileId)
        {
            if (!desired.Contains(tileId) || Failure.Length > 0) return false;
            staged.Add(tileId);
            return true;
        }

        public bool TryPublish()
        {
            if (Failure.Length > 0 || desired.Count == 0 || staged.Count != desired.Count) return false;
            foreach (string id in desired) if (!staged.Contains(id)) return false;
            // Complete detail replaces the global mesh.  The FAR object stays
            // resident as an inactive fallback, not as a second rendered soil.
            FarVisible = false; DetailVisible = true; return true;
        }

        public void Fail(string reason)
        {
            Failure = string.IsNullOrWhiteSpace(reason) ? "Detail tile load failed." : reason;
            staged.Clear(); FarVisible = true; DetailVisible = false;
        }
    }
}
