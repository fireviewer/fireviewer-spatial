using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;

namespace FireViewer.SpatialTiles
{
    public static class FwTileContainerDecoder
    {
        private static readonly byte[] Magic = { 0x46, 0x57, 0x54, 0x49, 0x4c, 0x45, 0x31, 0x00 };
        private const int PrefixBytes = 16;
        private const int MaximumHeaderBytes = 16 * 1024 * 1024;

        public static FwDecodedContainer Decode(byte[] payload, IFwJsonParser jsonParser)
        {
            if (payload == null || payload.Length < PrefixBytes || jsonParser == null)
                throw new ArgumentException("FWTile payload or JSON parser is missing.");
            for (int index = 0; index < Magic.Length; index++)
                if (payload[index] != Magic[index])
                    throw new FormatException("FWTile magic is invalid.");

            ushort major = ReadUInt16(payload, 8);
            ushort minor = ReadUInt16(payload, 10);
            int headerLength = checked((int)ReadUInt32(payload, 12));
            if (major != 1 || minor != 0)
                throw new FormatException($"Unsupported FWTile version {major}.{minor}.");
            if (headerLength <= 0 || headerLength > MaximumHeaderBytes || PrefixBytes + headerLength > payload.Length)
                throw new FormatException("FWTile header length is invalid.");

            string headerJson = Encoding.UTF8.GetString(payload, PrefixBytes, headerLength);
            FwContainerHeader header = jsonParser.Parse<FwContainerHeader>(headerJson);
            ValidateHeader(header);
            int bodyOffset = PrefixBytes + headerLength;
            var decodedSections = new Dictionary<string, byte[]>(StringComparer.Ordinal);
            foreach (FwSectionHeader section in header.sections)
            {
                if (!decodedSections.TryAdd(section.name, Array.Empty<byte>()))
                    throw new FormatException($"Duplicate FWTile section {section.name}.");
                if (section.offset_bytes < 0 || section.stored_bytes <= 0 || section.raw_bytes < 0 ||
                    bodyOffset + (long)section.offset_bytes + section.stored_bytes > payload.Length)
                    throw new FormatException($"FWTile section {section.name} range is invalid.");
                var stored = new byte[section.stored_bytes];
                Buffer.BlockCopy(payload, bodyOffset + section.offset_bytes, stored, 0, stored.Length);
                VerifySha256(stored, section.stored_sha256, $"{section.name} stored");
                if (!string.Equals(section.codec, "zlib", StringComparison.Ordinal))
                    throw new FormatException($"FWTile section {section.name} codec is unsupported.");
                byte[] raw = InflateZlib(stored, section.raw_bytes);
                VerifySha256(raw, section.raw_sha256, $"{section.name} raw");
                decodedSections[section.name] = raw;
            }
            return new FwDecodedContainer(header, decodedSections);
        }

        public static void VerifySha256(byte[] value, string expected, string label)
        {
            if (expected == null || expected.Length != 64)
                throw new FormatException($"{label} SHA-256 is invalid.");
            using SHA256 sha = SHA256.Create();
            byte[] digest = sha.ComputeHash(value);
            var text = new StringBuilder(64);
            foreach (byte item in digest)
                text.Append(item.ToString("x2"));
            if (!string.Equals(text.ToString(), expected, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"{label} SHA-256 mismatch.");
        }

        private static void ValidateHeader(FwContainerHeader header)
        {
            if (header == null || !string.Equals(header.schema, "fireviewer.fwtile.v1", StringComparison.Ordinal))
                throw new FormatException("FWTile header schema is unsupported.");
            if (!string.Equals(header.crs, "EPSG:2154", StringComparison.Ordinal) ||
                !string.Equals(header.linear_unit, "metre", StringComparison.Ordinal) ||
                !string.Equals(header.axis_convention, "X=east,Y=north,Z=up", StringComparison.Ordinal))
                throw new FormatException("FWTile spatial contract is invalid.");
            if (string.IsNullOrWhiteSpace(header.tile_id) || header.origin_l93_m == null || header.origin_l93_m.Length != 3 ||
                header.bounds_l93_m == null || header.bounds_l93_m.Length != 4 || header.sections == null || header.sections.Length == 0)
                throw new FormatException("FWTile identity, bounds, origin or sections are incomplete.");
        }

        private static byte[] InflateZlib(byte[] stored, int expectedRawBytes)
        {
            if (stored.Length < 6 || expectedRawBytes < 0)
                throw new InvalidDataException("Zlib stream is too short.");
            int cmf = stored[0];
            int flg = stored[1];
            if ((cmf & 0x0f) != 8 || (cmf >> 4) > 7 || ((cmf << 8) + flg) % 31 != 0 || (flg & 0x20) != 0)
                throw new InvalidDataException("Zlib header is unsupported or corrupt.");
            byte[] raw;
            using (var input = new MemoryStream(stored, 2, stored.Length - 6, false))
            using (var inflater = new DeflateStream(input, CompressionMode.Decompress, false))
            using (var output = new MemoryStream(expectedRawBytes))
            {
                var buffer = new byte[81920];
                int count;
                while ((count = inflater.Read(buffer, 0, buffer.Length)) > 0)
                {
                    if (output.Length + count > expectedRawBytes)
                        throw new InvalidDataException("Zlib stream expands beyond its declared size.");
                    output.Write(buffer, 0, count);
                }
                raw = output.ToArray();
            }
            if (raw.Length != expectedRawBytes)
                throw new InvalidDataException("Zlib stream does not match its declared size.");
            uint expectedAdler = ((uint)stored[^4] << 24) | ((uint)stored[^3] << 16) | ((uint)stored[^2] << 8) | stored[^1];
            if (Adler32(raw) != expectedAdler)
                throw new InvalidDataException("Zlib Adler-32 mismatch.");
            return raw;
        }

        private static uint Adler32(byte[] value)
        {
            const uint modulus = 65521;
            uint a = 1;
            uint b = 0;
            foreach (byte item in value)
            {
                a = (a + item) % modulus;
                b = (b + a) % modulus;
            }
            return (b << 16) | a;
        }

        internal static ushort ReadUInt16(byte[] value, int offset) =>
            (ushort)(value[offset] | value[offset + 1] << 8);

        internal static uint ReadUInt32(byte[] value, int offset) =>
            (uint)(value[offset] | value[offset + 1] << 8 | value[offset + 2] << 16 | value[offset + 3] << 24);

        internal static int ReadInt32(byte[] value, int offset) => unchecked((int)ReadUInt32(value, offset));
    }
}
