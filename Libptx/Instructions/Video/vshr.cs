using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.Video
{
    [Ptxop20("vshr.dtype.atype.u32{.sat}{.mode} d, a{.asel}, b{.bsel};")]
    [Ptxop20("vshr.dtype.atype.u32{.sat}{.mode}.op2 d, a{.asel}, b{.bsel}, c;")]
    [Ptxop20("vshr.dtype.atype.u32{.sat}{.mode} d.dsel, a{.asel}, b{.bsel}, c;")]
    [DebuggerNonUserCode]
    public partial class vshr : ptxop
    {
        [Affix] public Type dtype { get; set; }
        [Affix] public Type atype { get; set; }
        [Affix] public Type btype { get; set; }
        [Affix] public bool sat { get; set; }
        [Affix] public vshm mode { get; set; }
        [Affix] public op op2 { get; set; }

        protected override void custom_validate_opcode()
        {
            (dtype == s32 || dtype == u32).AssertTrue();
            (atype == s32 || atype == u32).AssertTrue();
            (btype == u32).AssertTrue();
            (mode == 0 || mode == vshm_clamp || mode == vshm_wrap).AssertTrue();
            (op2 == 0 || op2 == add || op2 == min || op2 == max).AssertTrue();
        }

        public vshr() { 1.UpTo(4).ForEach(_ => Operands.Add(null)); }
        [Destination] public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }
        public Expression c { get { return Operands[3]; } set { Operands[3] = value; } }

        protected override void custom_validate_operands()
        {
            var datamerge = op2 == 0 && c != null;
            if (datamerge)
            {
                is_reg(d, dtype, sel.exact()).AssertTrue();
                is_alu(a, atype, sel).AssertTrue();
                is_alu(b, btype, sel).AssertTrue();
                is_alu(c, dtype).AssertTrue();
            }
            else
            {
                is_reg(d, dtype).AssertTrue();
                is_alu(a, atype, sel).AssertTrue();
                is_alu(b, btype, sel).AssertTrue();
                is_alu_or_null(c, dtype).AssertTrue();
            }
        }
    }
}
