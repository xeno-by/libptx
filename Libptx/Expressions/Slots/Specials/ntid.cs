using System.Diagnostics;
using Libcuda.DataTypes;
using Libptx.Expressions.Slots.Specials.Annotations;

namespace Libptx.Expressions.Slots.Specials
{
    [Special("%ntid", typeof(uint4))]
    [DebuggerNonUserCode]
    public partial class ntid : Special
    {
    }
}

