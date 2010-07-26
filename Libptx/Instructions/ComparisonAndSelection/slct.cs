using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Instructions.Annotations;
using Libptx.Expressions;
using XenoGears.Assertions;
using Type=Libptx.Common.Types.Type;
using XenoGears.Functional;

namespace Libptx.Instructions.ComparisonAndSelection
{
    [Ptxop("slct.dtype.s32          d, a, b, c;")]
    [Ptxop("slct{.ftz}.dtype.f32    d, a, b, c;")]
    [DebuggerNonUserCode]
    public partial class slct : ptxop
    {
        [Affix] public bool ftz { get; set; }
        [Affix] public Type dtype { get; set; }
        [Affix] public Type ctype { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }

        protected override void custom_validate_opcode(Module ctx)
        {
            (ftz == true).AssertImplies(ctype == f32);
            (ctype == s32 || ctype == f32).AssertTrue();
        }

        public slct() { 1.UpTo(4).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }
        public Expression c { get { return Operands[3]; } set { Operands[3] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, dtype).AssertTrue();
            agree(a, dtype).AssertTrue();
            agree(b, dtype).AssertTrue();
            agree(c, ctype).AssertTrue();
        }
    }
}