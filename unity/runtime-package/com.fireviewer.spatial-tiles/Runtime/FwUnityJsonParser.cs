using System;
using UnityEngine;

namespace FireViewer.SpatialTiles
{
    internal sealed class FwUnityJsonParser : IFwJsonParser
    {
        public T Parse<T>(string json)
        {
            T value = JsonUtility.FromJson<T>(json);
            if (value == null) throw new FormatException($"JSON produced no {typeof(T).Name} object.");
            return value;
        }
    }
}
