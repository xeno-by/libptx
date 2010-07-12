using Libptx.Expressions;
using Libptx.Specials.Annotations;

namespace Libptx.Specials
{
    [Special20("%clock64", typeof(ulong))]
    public class Clock64 : Special
    {
    }
}