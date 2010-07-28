using System.Diagnostics;
using Libptx.Common.Annotations.Atoms;
using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Types.Opaques
{
    [Atom15]
    [DebuggerNonUserCode]
    public struct Texref
    {
        [Affix15("width")] public int Width { get; set; }
        [Affix15("height")] public int Height { get; set; }
        [Affix15("depth")] public int Depth { get; set; }
        [Affix15("channel_data_type")] public int ChannelDataType { get; set; }
        [Affix15("channel_order")] public int ChannelOrder { get; set; }
        [Affix15("normalized_coords")] public bool NormalizedCoords { get; set; }
        [Affix15("filter_mode")] public FilterMode FilterMode { get; set; }
        [Affix15("addr_mode_0")] public AddrMode AddrMode0 { get; set; }
        [Affix15("addr_mode_1")] public AddrMode AddrMode1 { get; set; }
        [Affix15("addr_mode_2")] public AddrMode AddrMode2 { get; set; }
    }
}
