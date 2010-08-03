using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.ComparisonAndSelection
{
    [Ptxop("set.cmpop{.ftz}.dtype.stype         d, a, b;")]
    [Ptxop("set.cmpop.boolop{.ftz}.dtype.stype  d, a, b, {!}c;")]
    [DebuggerNonUserCode]
    public partial class set : ptxop
    {
        [Affix] public cmp cmpop { get; set; }
        [Affix] public op boolop { get; set; }
        [Affix] public bool ftz { get; set; }
        [Affix] public Type dtype { get; set; }
        [Affix] public Type stype { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override void custom_validate_opcode()
        {
            (boolop == 0 || boolop == and || boolop == or || boolop == xor).AssertTrue();
            (ftz == true).AssertImplies(stype == f32);
            (dtype == u32 || dtype == s32 || dtype == f32).AssertTrue();
        }

        public set() { 1.UpTo(4).ForEach(_ => Operands.Add(null)); }
        [Destination] public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }
        public Expression c { get { return Operands[3]; } set { Operands[3] = value; } }

        protected override void custom_validate_operands()
        {
            is_reg(d, dtype).AssertTrue();
            is_alu(a, stype).AssertTrue();
            is_alu(b, stype).AssertTrue();
            is_alu_or_null(c, pred, not).AssertTrue();
        }
    }
}