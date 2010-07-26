using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Types.Opaques
{
    [DebuggerNonUserCode]
    public struct Samplerref
    {
        [Attr15("filter_mode")] public FilterMode FilterMode { get; set; }
        [Attr15("addr_mode_0")] public AddrMode AddrMode0 { get; set; }
        [Attr15("addr_mode_1")] public AddrMode AddrMode1 { get; set; }
        [Attr15("addr_mode_2")] public AddrMode AddrMode2 { get; set; }
    }
}