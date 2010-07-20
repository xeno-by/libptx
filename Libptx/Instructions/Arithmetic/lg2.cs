using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("lg2.approx{.ftz}.f32 d, a;")]
    [DebuggerNonUserCode]
    public class lg2 : ptxop
    {
        [Affix(SoftwareIsa.PTX_14)] public bool approx { get; set; }
        [Affix(SoftwareIsa.PTX_14)] public bool ftz { get; set; }
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            approx.AssertTrue();
            (ftz == true).AssertImplies(type == f32);
            (type == f32).AssertTrue();

            (target_swisa >= SoftwareIsa.PTX_14 && type == f64).AssertImplies(approx);
        }
    }
}