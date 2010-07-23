using Libptx.Expressions;
using Libptx.Expressions.Specials.Annotations;

namespace Libptx.Expressions.Specials
{
    [Special("%clock", typeof(uint))]
    public partial class clock32 : Special
    {
    }
}