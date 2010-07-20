using Libcuda.Versions;
using Libptx.Expressions;
using Libptx.Specials.Annotations;

namespace Libptx.Specials
{
    [Special("%warpid", typeof(uint), SoftwareIsa.PTX_13)]
    public class warpid : Special
    {
    }
}