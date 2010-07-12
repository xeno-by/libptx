using Libcuda.Versions;
using Libptx.Expressions;
using Libptx.Specials.Annotations;

namespace Libptx.Specials
{
    [Special10("%laneid", typeof(uint), SoftwareIsa.PTX_13)]
    public class Laneid : Special
    {
    }
}