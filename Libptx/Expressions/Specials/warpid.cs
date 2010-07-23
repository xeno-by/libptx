using Libcuda.Versions;
using Libptx.Expressions;
using Libptx.Expressions.Specials.Annotations;

namespace Libptx.Expressions.Specials
{
    [Special("%warpid", typeof(uint), SoftwareIsa.PTX_13)]
    public partial class warpid : Special
    {
    }
}