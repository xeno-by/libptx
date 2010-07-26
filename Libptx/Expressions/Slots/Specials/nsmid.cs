using System.Diagnostics;
using Libptx.Expressions.Slots.Specials.Annotations;

namespace Libptx.Expressions.Slots.Specials
{
    [Special20("%nsmid", typeof(uint))]
    [DebuggerNonUserCode]
    public partial class nsmid : Special
    {
    }
}