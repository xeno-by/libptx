using Libcuda.Versions;
using Libptx.Expressions;
using Libptx.Specials.Annotations;

namespace Libptx.Specials
{
    [Special10("%warpid", typeof(uint), SoftwareIsa.PTX_13)]
    public class Warpid : Special
    {
    }
}