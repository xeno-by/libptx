using Libcuda.DataTypes;
using Libptx.Expressions;
using Libptx.Specials.Annotations;

namespace Libptx.Specials
{
    [Special("%ctaid", typeof(uint4))]
    public class Ctaid : Special
    {
    }
}