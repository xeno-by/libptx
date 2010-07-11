using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("rsqrt.approx{.ftz}.f32  d, a;")]
    [Ptxop("rsqrt.approx.f64        d, a;")]
    [DebuggerNonUserCode]
    internal class rsqrt : ptxop
    {
        [Suffix(SoftwareIsa.PTX_14)] public bool approx { get; set; }
        [Suffix(SoftwareIsa.PTX_14)] public bool ftz { get; set; }
        [Suffix] public type type { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            approx.AssertTrue();
            (ftz == true).AssertImplies(type == f32);
            type.isfloat().AssertTrue();

            (target_swisa >= SoftwareIsa.PTX_14 && type == f64).AssertImplies(approx);
        }
    }
}