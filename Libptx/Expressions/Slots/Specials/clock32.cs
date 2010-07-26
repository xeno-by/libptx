using Libptx.Expressions.Slots.Specials.Annotations;

namespace Libptx.Expressions.Slots.Specials
{
    [Special("%clock", typeof(uint))]
    public partial class clock32 : Special
    {
    }
}