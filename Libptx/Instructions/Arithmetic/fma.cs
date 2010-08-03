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
    [Ptxop("fma.rnd{.ftz}{.sat}.f32 d, a, b, c;")]
    [Ptxop("fma.rnd.f64             d, a, b, c;")]
    [DebuggerNonUserCode]
    public partial class fma : ptxop
    {
        [Affix] public frnd rnd { get; set; }
        [Affix] public bool ftz { get; set; }
        [Affix] public bool sat { get; set; }
        [Affix] public Type type { get; set; }

        protected override SoftwareIsa custom_swisa
        {
            get { return type == f32 ? SoftwareIsa.PTX_20 : SoftwareIsa.PTX_10; }
        }

        protected override HardwareIsa custom_hwisa
        {
            get { return type == f32 ? HardwareIsa.SM_20 : HardwareIsa.SM_10; }
        }

        protected override void custom_validate_opcode()
        {
            (rnd != 0).AssertTrue();
            (ftz == true).AssertImplies(type == f32);
            (sat == true).AssertImplies(type == f32);
            type.is_float().AssertTrue();
        }

        public fma() { 1.UpTo(4).ForEach(_ => Operands.Add(null)); }
        [Destination] public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }
        public Expression c { get { return Operands[3]; } set { Operands[3] = value; } }

        protected override void custom_validate_operands()
        {
            is_reg(d, type).AssertTrue();
            is_alu(a, type).AssertTrue();
            is_alu(b, type).AssertTrue();
            is_alu(c, type).AssertTrue();
        }
    }
}