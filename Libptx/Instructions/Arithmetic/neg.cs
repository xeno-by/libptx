using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("neg.type        d, a;")]
    [Ptxop("neg{.ftz}.f32   d, a;")]
    [Ptxop("neg.f64         d, a;")]
    [DebuggerNonUserCode]
    internal class neg : ptxop
    {
        [Suffix] public bool ftz { get; set; }
        [Suffix] public type type { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (ftz == true).AssertImplies(type == f32);
            (type.issigned() || type.isfloat()).AssertTrue();
        }
    }
}