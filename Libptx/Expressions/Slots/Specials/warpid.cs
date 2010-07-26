using Libcuda.Versions;
using Libptx.Expressions.Slots.Specials.Annotations;

namespace Libptx.Expressions.Slots.Specials
{
    [Special("%warpid", typeof(uint), SoftwareIsa.PTX_13)]
    public partial class warpid : Special
    {
    }
}