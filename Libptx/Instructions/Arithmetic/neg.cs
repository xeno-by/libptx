using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop10("neg.type        d, a;")]
    [Ptxop10("neg{.ftz}.f32   d, a;")]
    [Ptxop10("neg.f64         d, a;")]
    [DebuggerNonUserCode]
    internal class neg : ptxop
    {
        [Infix] public bool ftz { get; set; }
        [Infix] public type type { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (ftz == true).AssertImplies(type == f32);
            (type.issigned() || type.isfloat()).AssertTrue();
        }
    }
}