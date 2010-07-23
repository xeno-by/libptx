using Libcuda.Versions;
using Libptx.Expressions;
using Libptx.Expressions.Specials.Annotations;

namespace Libptx.Expressions.Specials
{
    [Special("%laneid", typeof(uint), SoftwareIsa.PTX_13)]
    public partial class laneid : Special
    {
    }
}