using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Expressions.Slots.Specials.Annotations;
using XenoGears.Assertions;

namespace Libptx.Expressions.Slots.Specials
{
    [Special("%envreg{index}", typeof(Bit32), SoftwareIsa.PTX_21, HardwareIsa.SM_10)]
    public partial class envreg : Special
    {
        [Affix("index")] public int Index { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            (0 <= Index && Index <= 31).AssertTrue();
        }
    }
}