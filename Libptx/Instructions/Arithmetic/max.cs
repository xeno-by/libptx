using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("max.type        d, a, b;")]
    [Ptxop("max{.ftz}.f32   d, a, b;")]
    [Ptxop("max.f64         d, a, b;")]
    [DebuggerNonUserCode]
    public class max : ptxop
    {
        [Affix] public bool ftz { get; set; }
        [Affix] public type type { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (ftz == true).AssertImplies(type == f32);
        }
    }
}