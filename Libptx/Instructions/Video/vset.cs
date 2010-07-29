using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.Video
{
    [Ptxop20("vset.atype.btype.cmp d, a{.asel}, b{.bsel};")]
    [Ptxop20("vset.atype.btype.cmp.op2 d, a{.asel}, b{.bsel}, c;")]
    [Ptxop20("vset.atype.btype.cmp d.dsel, a{.asel}, b{.bsel}, c;")]
    [DebuggerNonUserCode]
    public partial class vset : ptxop
    {
        [Affix] public Type atype { get; set; }
        [Affix] public Type btype { get; set; }
        [Affix] public cmp cmp { get; set; }
        [Affix] public op op2 { get; set; }

        protected override void custom_validate_opcode()
        {
            (atype == s32 || atype == u32).AssertTrue();
            (btype == s32 || btype == u32).AssertTrue();
            (cmp == eq || cmp == ne || cmp == lt || cmp == le || cmp == gt || cmp == ge).AssertTrue();
            (op2 == 0 || op2 == add || op2 == min || op2 == max).AssertTrue();
        }

        public vset() { 1.UpTo(4).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }
        public Expression c { get { return Operands[3]; } set { Operands[3] = value; } }

        protected override void custom_validate_operands()
        {
            var datamerge = op2 == 0 && c != null;
            if (datamerge)
            {
                is_reg(d, u32, sel.exact()).AssertTrue();
                is_alu(a, atype, sel).AssertTrue();
                is_alu(b, btype, sel).AssertTrue();
                is_alu(c, u32).AssertTrue();
            }
            else
            {
                is_reg(d, u32).AssertTrue();
                is_alu(a, atype, sel).AssertTrue();
                is_alu(b, btype, sel).AssertTrue();
                is_alu_or_null(c, u32).AssertTrue();
            }
        }
    }
}