using Libcuda.Versions;
using Libptx.Instructions.Annotations;

namespace Libptx.Instructions.Miscellaneous
{
    [Ptxop("pmevent a;", SoftwareIsa.PTX_14)]
    internal class pmevent : ptxop
    {
    }
}