using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Types.Opaques
{
    public enum AddrMode
    {
        [Affix15("wrap")] Wrap,
        [Affix15("mirror")] Mirror,
        [Affix15("clamp_ogl")] ClampOGL,
        [Affix15("clamp_to_edge")] ClampToEdge,
        [Affix15("clamp_to_border")] ClampToBorder,
    }
}