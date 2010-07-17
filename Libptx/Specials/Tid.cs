using Libcuda.DataTypes;
using Libptx.Expressions;
using Libptx.Specials.Annotations;

namespace Libptx.Specials
{
    [Special("%tid", typeof(uint4))]
    public class Tid : Special
    {
    }
}
