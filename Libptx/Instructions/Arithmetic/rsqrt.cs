using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop10("rsqrt.approx{.ftz}.f32  d, a;")]
    [Ptxop10("rsqrt.approx.f64        d, a;")]
    [DebuggerNonUserCode]
    internal class rsqrt : ptxop
    {
        [Infix(SoftwareIsa.PTX_14)] public bool approx { get; set; }
        [Infix(SoftwareIsa.PTX_14)] public bool ftz { get; set; }
        [Infix] public type type { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            approx.AssertTrue();
            (ftz == true).AssertImplies(type == f32);
            type.isfloat().AssertTrue();

            (target_swisa >= SoftwareIsa.PTX_14 && type == f64).AssertImplies(approx);
        }
    }
}