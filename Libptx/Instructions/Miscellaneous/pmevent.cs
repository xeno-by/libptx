using Libcuda.Versions;
using Libptx.Instructions.Annotations;

namespace Libptx.Instructions.Miscellaneous
{
    [Ptxop10("pmevent a;", SoftwareIsa.PTX_14)]
    internal class pmevent : ptxop
    {
    }
}