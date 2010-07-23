using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Expressions;
using Libptx.Expressions.Specials.Annotations;
using XenoGears.Assertions;

namespace Libptx.Expressions.Specials
{
    [Special("%pm{index}", typeof(uint), SoftwareIsa.PTX_13)]
    public partial class pm : Special
    {
        [Affix("index")] public int Index { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            (0 <= Index && Index <= 3).AssertTrue();
        }
    }
}