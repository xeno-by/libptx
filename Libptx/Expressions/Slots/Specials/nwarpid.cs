using System.Diagnostics;
using Libptx.Expressions.Slots.Specials.Annotations;

namespace Libptx.Expressions.Slots.Specials
{
    [Special20("%nwarpid", typeof(uint))]
    [DebuggerNonUserCode]
    public partial class nwarpid : Special
    {
    }
}