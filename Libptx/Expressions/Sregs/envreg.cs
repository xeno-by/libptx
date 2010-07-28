using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types.Bits;
using Libptx.Expressions.Sregs.Annotations;
using XenoGears.Assertions;

namespace Libptx.Expressions.Sregs
{
    [Sreg("%envreg{index}", typeof(Bit32), SoftwareIsa.PTX_21, HardwareIsa.SM_10)]
    [DebuggerNonUserCode]
    public partial class envreg : Sreg
    {
        [Affix("index")] public int Index { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            (0 <= Index && Index <= 31).AssertTrue();
        }
    }
}