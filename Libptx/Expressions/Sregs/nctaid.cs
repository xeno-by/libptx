using System.Diagnostics;
using Libcuda.DataTypes;
using Libptx.Expressions.Sregs.Annotations;

namespace Libptx.Expressions.Sregs
{
    [Special("%nctaid", typeof(uint4))]
    [DebuggerNonUserCode]
    public partial class nctaid : Sreg
    {
    }
}