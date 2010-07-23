using Libptx.Common.Enumerations;
using Libptx.Common.Annotations.Quanta;
using Libptx.Expressions;
using Libptx.Expressions.Specials.Annotations;
using XenoGears.Assertions;

namespace Libptx.Expressions.Specials
{
    [Special20("%lanemask_{op}", typeof(uint))]
    public partial class lanemask : Special
    {
        [Affix("op")] public cmp Mask { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            (Mask == eq || Mask == gt || Mask == ge || Mask == lt || Mask == le).AssertTrue();
        }
    }
}