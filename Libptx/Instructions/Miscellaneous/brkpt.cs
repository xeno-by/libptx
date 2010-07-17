using Libcuda.Versions;
using Libptx.Instructions.Annotations;

namespace Libptx.Instructions.Miscellaneous
{
    [Ptxop("brkpt;", HardwareIsa.SM_11)]
    public class brkpt : ptxop
    {
    }
}