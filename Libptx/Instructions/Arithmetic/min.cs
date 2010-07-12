using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop10("min.type        d, a, b;")]
    [Ptxop10("min{.ftz}.f32   d, a, b;")]
    [Ptxop10("min.f64         d, a, b;")]
    [DebuggerNonUserCode]
    internal class min : ptxop
    {
        [Infix] public bool ftz { get; set; }
        [Infix] public type type { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (ftz == true).AssertImplies(type == f32);
        }
    }
}