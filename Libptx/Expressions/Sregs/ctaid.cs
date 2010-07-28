using System.Diagnostics;
using Libcuda.DataTypes;
using Libptx.Expressions.Sregs.Annotations;

namespace Libptx.Expressions.Sregs
{
    [Sreg("%ctaid", typeof(uint4))]
    [DebuggerNonUserCode]
    public partial class ctaid : Sreg
    {
    }
}