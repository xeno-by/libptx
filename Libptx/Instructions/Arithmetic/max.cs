using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop10("max.type        d, a, b;")]
    [Ptxop10("max{.ftz}.f32   d, a, b;")]
    [Ptxop10("max.f64         d, a, b;")]
    [DebuggerNonUserCode]
    internal class max : ptxop
    {
        [Infix] public bool ftz { get; set; }
        [Infix] public type type { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (ftz == true).AssertImplies(type == f32);
        }
    }
}