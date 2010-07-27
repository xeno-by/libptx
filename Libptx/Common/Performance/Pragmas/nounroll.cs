using Libcuda.Versions;
using Libptx.Common.Performance.Pragmas.Annotations;

namespace Libptx.Common.Performance.Pragmas
{
    [Pragma("nounroll", SoftwareIsa.PTX_20, HardwareIsa.SM_10)]
    public class nounroll : Pragma
    {
    }
}