using System.Diagnostics;
using Libptx.Expressions.Slots.Specials.Annotations;

namespace Libptx.Expressions.Slots.Specials
{
    [Special("%clock", typeof(uint))]
    [DebuggerNonUserCode]
    public partial class clock32 : Special
    {
    }
}