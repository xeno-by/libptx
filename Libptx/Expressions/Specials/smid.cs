using Libcuda.Versions;
using Libptx.Expressions;
using Libptx.Expressions.Specials.Annotations;

namespace Libptx.Expressions.Specials
{
    [Special("%smid", typeof(uint), SoftwareIsa.PTX_13)]
    public partial class smid : Special
    {
    }
}