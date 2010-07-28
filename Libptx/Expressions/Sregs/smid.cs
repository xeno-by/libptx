using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Expressions.Sregs.Annotations;

namespace Libptx.Expressions.Sregs
{
    [Sreg("%smid", typeof(uint), SoftwareIsa.PTX_13)]
    [DebuggerNonUserCode]
    public partial class smid : Sreg
    {
    }
}