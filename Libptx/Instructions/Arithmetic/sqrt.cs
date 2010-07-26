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
    [Ptxop("sqrt.approx{.ftz}.f32    d, a;")]
    [Ptxop("sqrt.rnd{.ftz}.f32       d, a;")]
    [Ptxop("sqrt.rnd.f64             d, a;")]
    [DebuggerNonUserCode]
    public partial class sqrt : ptxop
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
                return rzmp ? SoftwareIsa.PTX_20 : SoftwareIsa.PTX_14;
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

        protected override void custom_validate_opcode(Module ctx)
        {
            (approx == true).AssertImplies(type == f32);
            (rnd != 0).AssertEquiv(!approx);
            (ftz == true).AssertImplies(type == f32);
            type.is_float().AssertTrue();

            (ctx.Version >= SoftwareIsa.PTX_14 && type == f64).AssertImplies(approx || rnd != 0);
        }

        public sqrt() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, type).AssertTrue();
            agree(a, type).AssertTrue();
        }
    }
}