using Libcuda.Versions;
using Libptx.Expressions;
using Libptx.Specials.Annotations;

namespace Libptx.Specials
{
    [Special("%laneid", typeof(uint), SoftwareIsa.PTX_13)]
    public class laneid : Special
    {
    }
}