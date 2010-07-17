using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Expressions;
using Libptx.Specials.Annotations;
using XenoGears.Assertions;

namespace Libptx.Specials
{
    [Special("%envreg{index}", typeof(Bit32), SoftwareIsa.PTX_21, HardwareIsa.SM_10)]
    public class Envreg : Special
    {
        [Affix("index")] public int Index { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            (0 <= Index && Index <= 31).AssertTrue();
        }
    }
}