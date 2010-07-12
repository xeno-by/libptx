using Libcuda.Versions;
using Libptx.Expressions;
using Libptx.Specials.Annotations;

namespace Libptx.Specials
{
    [Special10("%smid", typeof(uint), SoftwareIsa.PTX_13)]
    public class Smid : Special
    {
    }
}