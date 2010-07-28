using System.Diagnostics;
using Libptx.Common.Annotations.Atoms;
using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Types.Opaques
{
    [Atom15]
    [DebuggerNonUserCode]
    public struct Surfref
    {
        [Affix15("width")] public int Width { get; set; }
        [Affix15("height")] public int Height { get; set; }
        [Affix15("depth")] public int Depth { get; set; }
        [Affix15("channel_data_type")] public int ChannelDataType { get; set; }
        [Affix15("channel_order")] public int ChannelOrder { get; set; }
    }
}