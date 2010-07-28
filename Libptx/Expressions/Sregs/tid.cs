using System.Diagnostics;
using Libcuda.DataTypes;
using Libptx.Expressions.Sregs.Annotations;

namespace Libptx.Expressions.Sregs
{
    [Sreg("%tid", typeof(uint4))]
    [DebuggerNonUserCode]
    public partial class tid : Sreg
    {
    }
}
