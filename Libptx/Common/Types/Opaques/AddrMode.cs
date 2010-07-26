using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Types.Opaques
{
    public enum AddrMode
    {
        [Attr15("wrap")] Wrap,
        [Attr15("mirror")] Mirror,
        [Attr15("clamp_ogl")] ClampOGL,
        [Attr15("clamp_to_edge")] ClampToEdge,
        [Attr15("clamp_to_border")] ClampToBorder,
    }
}