using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using Type=Libptx.Common.Types.Type;
using XenoGears.Functional;

namespace Libptx.Instructions.ComparisonAndSelection
{
    [Ptxop("setp.cmpop{.ftz}.type              p[|q], a, b;")]
    [Ptxop("setp.cmpop.boolop{.ftz}.type       p[|q], a, b, {!}c;")]
    [DebuggerNonUserCode]
    public partial class setp : ptxop
    {
        [Affix] public cmp cmpop { get; set; }
        [Affix] public op boolop { get; set; }
        [Affix] public bool ftz { get; set; }
        [Affix] public Type type { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override void custom_validate_opcode(Module ctx)
        {
            (boolop == 0 || boolop == and || boolop == or || boolop == xor).AssertTrue();
            (ftz == true).AssertImplies(type == f32);
        }

        setp() { 1.UpTo(4).ForEach(_ => Operands.Add(null)); }
        public Expression p { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }
        public Expression c { get { return Operands[3]; } set { Operands[3] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(p, pred, couple).AssertTrue();
            agree(a, type).AssertTrue();
            agree(b, type).AssertTrue();
            agree_or_null(c, pred, not).AssertTrue();
        }
    }
}