using Libcuda.DataTypes;
using Libptx.Expressions;
using Libptx.Specials.Annotations;

namespace Libptx.Specials
{
    [Special("%nctaid", typeof(uint4))]
    public class nctaid : Special
    {
    }
}