using Libcuda.DataTypes;
using Libptx.Expressions;
using Libptx.Expressions.Specials.Annotations;

namespace Libptx.Expressions.Specials
{
    [Special("%ntid", typeof(uint4))]
    public partial class ntid : Special
    {
    }
}

