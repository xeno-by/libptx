using System.Diagnostics;
using Libptx.Expressions.Slots.Specials.Annotations;

namespace Libptx.Expressions.Slots.Specials
{
    [Special20("%clock64", typeof(ulong))]
    [DebuggerNonUserCode]
    public partial class clock64 : Special
    {
    }
}