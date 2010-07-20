using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Edsl.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("div.type                d, a, b;")]
    [Ptxop("div.approx{.ftz}.f32    d, a, b;")]
    [Ptxop("div.full{.ftz}.f32      d, a, b;")]
    [Ptxop("div.rnd{.ftz}.f32       d, a, b;")]
    [Ptxop("div.rnd.f64             d, a, b;")]
    [DebuggerNonUserCode]
    public class div : ptxop
    {
        [Affix(SoftwareIsa.PTX_14)] public bool approx { get; set; }
        [Affix(SoftwareIsa.PTX_14)] public bool full { get; set; }
        [Affix(SoftwareIsa.PTX_14)] public frnd rnd { get; set; }
        [Affix(SoftwareIsa.PTX_14)] public bool ftz { get; set; }
        [Affix] public Type type { get; set; }

        protected override HardwareIsa custom_hwisa
        {
            get
            {
                var f32_rnd = type == f32 && rnd != 0;
                var f64_rzmp = type == f64 && (rnd == rz || rnd == rm || rnd == rp);
                return (f32_rnd || f64_rzmp) ? HardwareIsa.SM_20 : HardwareIsa.SM_10;
            }
        }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (approx == true).AssertImplies(type == f32);
            (full == true).AssertImplies(type == f32);
            (approx && full).AssertFalse();
            (rnd != 0).AssertEquiv(!approx && !full && type.isfloat());
            (ftz == true).AssertImplies(type == f32);

            (target_swisa >= SoftwareIsa.PTX_14 && type == f64).AssertImplies(approx || full || rnd != 0);
        }
    }
}