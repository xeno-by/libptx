using Libcuda.DataTypes;
using Libptx.Expressions;
using Libptx.Specials.Annotations;

namespace Libptx.Specials
{
    [Special("%ntid", typeof(uint4))]
    public class ntid : Special
    {
    }
}

