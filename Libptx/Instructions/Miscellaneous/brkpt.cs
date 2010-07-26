using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations;

namespace Libptx.Instructions.Miscellaneous
{
    [Ptxop("brkpt;", HardwareIsa.SM_11)]
    [DebuggerNonUserCode]
    public partial class brkpt : ptxop
    {
    }
}