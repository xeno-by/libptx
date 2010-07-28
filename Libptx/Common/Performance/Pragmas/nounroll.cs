using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Performance.Pragmas.Annotations;

namespace Libptx.Common.Performance.Pragmas
{
    [Pragma("nounroll", SoftwareIsa.PTX_20, HardwareIsa.SM_10)]
    [DebuggerNonUserCode]
    public class nounroll : Pragma
    {
    }
}