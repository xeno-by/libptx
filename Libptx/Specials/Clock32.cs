using Libptx.Expressions;
using Libptx.Specials.Annotations;

namespace Libptx.Specials
{
    [Special("%clock", typeof(uint))]
    public class clock32 : Special
    {
    }
}