using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Common.Types;
using Libptx.Expressions;
using Libptx.Specials.Annotations;
using XenoGears.Assertions;

namespace Libptx.Specials
{
    [Special10("%envreg{index}", typeof(Bit32), SoftwareIsa.PTX_21, HardwareIsa.SM_10)]
    public class Envreg : Special
    {
        [Infix("index")] public int Index { get; set; }

        public override void Validate()
        {
            (0 <= Index && Index <= 31).AssertTrue();
        }
    }
}