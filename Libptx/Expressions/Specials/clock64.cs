using Libptx.Expressions;
using Libptx.Expressions.Specials.Annotations;

namespace Libptx.Expressions.Specials
{
    [Special20("%clock64", typeof(ulong))]
    public partial class clock64 : Special
    {
    }
}