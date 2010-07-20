using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("rcp.approx{.ftz}.f32    d, a;")]
    [Ptxop("rcp.approx{.ftz}.f64    d, a;")]
    [Ptxop("rcp.rnd{.ftz}.f32       d, a;")]
    [Ptxop("rcp.rnd.f64             d, a;")]
    [DebuggerNonUserCode]
    public class rcp : ptxop
    {
        [Affix(SoftwareIsa.PTX_14)] public bool approx { get; set; }
        [Affix(SoftwareIsa.PTX_14)] public frnd rnd { get; set; }
        [Affix(SoftwareIsa.PTX_14)] public bool ftz { get; set; }
        [Affix] public type type { get; set; }

        protected override SoftwareIsa custom_swisa
        {
            get
            {
                var rzmp = rnd == rz || rnd == rm || rnd == rp;
                if (rzmp) return SoftwareIsa.PTX_20;

                var approx_ftz_f64 = approx && ftz && type == f64;
                return approx_ftz_f64 ? SoftwareIsa.PTX_21 : SoftwareIsa.PTX_10;
            }
        }

        protected override HardwareIsa custom_hwisa
        {
            get
            {
                var f32_rnd = type == f32 && rnd != null;
                var f64_rzmp = type == f64 && (rnd == rz || rnd == rm || rnd == rp);
                return (f32_rnd || f64_rzmp) ? HardwareIsa.SM_20 : HardwareIsa.SM_10;
            }
        }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (rnd != null).AssertEquiv(!approx);
            (rnd != null && type == f64).AssertImplies(approx == false);
            type.isfloat().AssertTrue();

            (target_swisa >= SoftwareIsa.PTX_14 && type == f64).AssertImplies(approx || rnd != null);
        }
    }
}