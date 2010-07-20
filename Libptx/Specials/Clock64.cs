using Libptx.Expressions;
using Libptx.Specials.Annotations;

namespace Libptx.Specials
{
    [Special20("%clock64", typeof(ulong))]
    public class clock64 : Special
    {
    }
}