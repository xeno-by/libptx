using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("rcp.approx{.ftz}.f32    d, a;")]
    [Ptxop("rcp.approx{.ftz}.f64    d, a;")]
    [Ptxop("rcp.rnd{.ftz}.f32       d, a;")]
    [Ptxop("rcp.rnd.f64             d, a;")]
    [DebuggerNonUserCode]
    public partial class rcp : ptxop
    {
        [Affix(SoftwareIsa.PTX_14)] public bool approx { get; set; }
        [Affix(SoftwareIsa.PTX_14)] public frnd rnd { get; set; }
        [Affix(SoftwareIsa.PTX_14)] public bool ftz { get; set; }
        [Affix] public Type type { get; set; }

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
                var f32_rnd = type == f32 && rnd != 0;
                var f64_rzmp = type == f64 && (rnd == rz || rnd == rm || rnd == rp);
                return (f32_rnd || f64_rzmp) ? HardwareIsa.SM_20 : HardwareIsa.SM_10;
            }
        }

        protected override void custom_validate_opcode()
        {
            (rnd != 0).AssertEquiv(!approx);
            (rnd != 0 && type == f64).AssertImplies(approx == false);
            (rnd != 0 && type == f64).AssertImplies(ftz == false);
            type.is_float().AssertTrue();

            (ctx.Version >= SoftwareIsa.PTX_14 && type == f64).AssertImplies(approx || rnd != 0);
        }

        public rcp() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands()
        {
            is_reg(d, type).AssertTrue();
            is_alu(a, type).AssertTrue();
        }
    }
}