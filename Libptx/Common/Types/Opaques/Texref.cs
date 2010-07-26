using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Types.Opaques
{
    [DebuggerNonUserCode]
    public struct Texref
    {
        [Attr15("width")] public int Width { get; set; }
        [Attr15("height")] public int Height { get; set; }
        [Attr15("depth")] public int Depth { get; set; }
        [Attr15("channel_data_type")] public int ChannelDataType { get; set; }
        [Attr15("channel_order")] public int ChannelOrder { get; set; }
        [Attr15("normalized_coords")] public bool NormalizedCoords { get; set; }
        [Attr15("filter_mode")] public FilterMode FilterMode { get; set; }
        [Attr15("addr_mode_0")] public AddrMode AddrMode0 { get; set; }
        [Attr15("addr_mode_1")] public AddrMode AddrMode1 { get; set; }
        [Attr15("addr_mode_2")] public AddrMode AddrMode2 { get; set; }
    }
}
