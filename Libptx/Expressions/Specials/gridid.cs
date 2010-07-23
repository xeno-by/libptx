using Libptx.Expressions;
using Libptx.Expressions.Specials.Annotations;

namespace Libptx.Expressions.Specials
{
    [Special("%gridid", typeof(uint))]
    public partial class gridid : Special
    {
    }
}