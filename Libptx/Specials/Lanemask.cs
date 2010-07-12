using Libptx.Common.Enumerations;
using Libptx.Common.Infrastructure;
using Libptx.Expressions;
using Libptx.Specials.Annotations;
using XenoGears.Assertions;

namespace Libptx.Specials
{
    [Special20("%lanemask_{op}", typeof(uint))]
    public class Lanemask : Special
    {
        [Infix("op")] public Comparison Mask { get; set; }

        public override void Validate()
        {
            (Mask == eq || Mask == gt || Mask == ge || Mask == lt || Mask == le).AssertTrue();
        }
    }
}