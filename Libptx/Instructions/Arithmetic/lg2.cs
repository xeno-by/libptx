using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("lg2.approx{.ftz}.f32 d, a;")]
    [DebuggerNonUserCode]
    internal class lg2 : ptxop
    {
        [Suffix(SoftwareIsa.PTX_14)] public bool approx { get; set; }
        [Suffix(SoftwareIsa.PTX_14)] public bool ftz { get; set; }
        [Suffix] public type type { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            approx.AssertTrue();
            (ftz == true).AssertImplies(type == f32);
            (type == f32).AssertTrue();

            (target_swisa >= SoftwareIsa.PTX_14 && type == f64).AssertImplies(approx);
        }
    }
}