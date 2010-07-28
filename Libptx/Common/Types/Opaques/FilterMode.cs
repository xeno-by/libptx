using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Types.Opaques
{
    public enum FilterMode
    {
        [Affix15("nearest")] Nearest,
        [Affix15("linear")] Linear,
    }
}