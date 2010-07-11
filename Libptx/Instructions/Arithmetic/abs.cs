using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("abs.type        d, a;")]
    [Ptxop("abs{.ftz}.f32   d, a;")]
    [Ptxop("abs.f64         d, a;")]
    [DebuggerNonUserCode]
    internal class abs : ptxop
    {
        [Suffix] public bool ftz { get; set; }
        [Suffix] public type type { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (type.issigned() || type.isfloat()).AssertTrue();
            (ftz == true).AssertImplies(type == f32);
        }
    }
}