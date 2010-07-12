using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop10("abs.type        d, a;")]
    [Ptxop10("abs{.ftz}.f32   d, a;")]
    [Ptxop10("abs.f64         d, a;")]
    [DebuggerNonUserCode]
    internal class abs : ptxop
    {
        [Infix] public bool ftz { get; set; }
        [Infix] public type type { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (type.issigned() || type.isfloat()).AssertTrue();
            (ftz == true).AssertImplies(type == f32);
        }
    }
}