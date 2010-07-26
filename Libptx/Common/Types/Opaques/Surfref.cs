using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Types.Opaques
{
    [DebuggerNonUserCode]
    public struct Surfref
    {
        [Attr15("width")] public int Width { get; set; }
        [Attr15("height")] public int Height { get; set; }
        [Attr15("depth")] public int Depth { get; set; }
        [Attr15("channel_data_type")] public int ChannelDataType { get; set; }
        [Attr15("channel_order")] public int ChannelOrder { get; set; }
    }
}