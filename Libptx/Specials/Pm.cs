using Libcuda.Versions;
using Libptx.Common.Annotations.Quantas;
using Libptx.Expressions;
using Libptx.Specials.Annotations;
using XenoGears.Assertions;

namespace Libptx.Specials
{
    [Special("%pm{index}", typeof(uint), SoftwareIsa.PTX_13)]
    public class Pm : Special
    {
        [Affix("index")] public int Index { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            (0 <= Index && Index <= 3).AssertTrue();
        }
    }
}