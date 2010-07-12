using Libcuda.Versions;
using Libptx.Instructions.Annotations;

namespace Libptx.Instructions.Miscellaneous
{
    [Ptxop10("brkpt;", HardwareIsa.SM_11)]
    internal class brkpt : ptxop
    {
    }
}