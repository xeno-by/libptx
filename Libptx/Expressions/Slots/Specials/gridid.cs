using System.Diagnostics;
using Libptx.Expressions.Slots.Specials.Annotations;

namespace Libptx.Expressions.Slots.Specials
{
    [Special("%gridid", typeof(uint))]
    [DebuggerNonUserCode]
    public partial class gridid : Special
    {
    }
}