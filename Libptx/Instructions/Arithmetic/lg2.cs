using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop10("lg2.approx{.ftz}.f32 d, a;")]
    [DebuggerNonUserCode]
    internal class lg2 : ptxop
    {
        [Infix(SoftwareIsa.PTX_14)] public bool approx { get; set; }
        [Infix(SoftwareIsa.PTX_14)] public bool ftz { get; set; }
        [Infix] public type type { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            approx.AssertTrue();
            (ftz == true).AssertImplies(type == f32);
            (type == f32).AssertTrue();

            (target_swisa >= SoftwareIsa.PTX_14 && type == f64).AssertImplies(approx);
        }
    }
}