using System.Diagnostics;
using Libcuda.DataTypes;
using Libptx.Expressions.Sregs.Annotations;

namespace Libptx.Expressions.Sregs
{
    [Sreg("%ntid", typeof(uint4))]
    [DebuggerNonUserCode]
    public partial class ntid : Sreg
    {
    }
}

