using System.Diagnostics;
using Libcuda.DataTypes;
using Libptx.Expressions.Slots.Specials.Annotations;

namespace Libptx.Expressions.Slots.Specials
{
    [Special("%ctaid", typeof(uint4))]
    [DebuggerNonUserCode]
    public partial class ctaid : Special
    {
    }
}