using System.Diagnostics;
using Libcuda.DataTypes;
using Libptx.Expressions.Slots.Specials.Annotations;

namespace Libptx.Expressions.Slots.Specials
{
    [Special("%nctaid", typeof(uint4))]
    [DebuggerNonUserCode]
    public partial class nctaid : Special
    {
    }
}