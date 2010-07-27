using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Expressions.Sregs.Annotations;

namespace Libptx.Expressions.Sregs
{
    [Special("%laneid", typeof(uint), SoftwareIsa.PTX_13)]
    [DebuggerNonUserCode]
    public partial class laneid : Sreg
    {
    }
}