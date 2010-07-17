using Libptx.Common.Enumerations;
using Libptx.Common.Annotations.Quanta;
using Libptx.Expressions;
using Libptx.Specials.Annotations;
using XenoGears.Assertions;

namespace Libptx.Specials
{
    [Special20("%lanemask_{op}", typeof(uint))]
    public class Lanemask : Special
    {
        [Affix("op")] public Comparison Mask { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            (Mask == eq || Mask == gt || Mask == ge || Mask == lt || Mask == le).AssertTrue();
        }
    }
}