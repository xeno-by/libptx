using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Types.Opaques
{
    public enum FilterMode
    {
        [Attr15("nearest")] Nearest,
        [Attr15("linear")] Linear,
    }
}