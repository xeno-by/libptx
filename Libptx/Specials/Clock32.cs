using Libptx.Expressions;
using Libptx.Specials.Annotations;

namespace Libptx.Specials
{
    [Special("%clock", typeof(uint))]
    public class Clock32 : Special
    {
    }
}