using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop20("sured.p.op.geom.ctype.clampm [a,b],c;")]
    [DebuggerNonUserCode]
    public partial class sured_p : ptxop
    {
        [Affix] public op op { get; set; }
        [Affix] public geom geom { get; set; }
        [Affix] public Type ctype { get; set; }
        [Affix] public clampm clampm { get; set; }

        protected override bool allow_bit32 { get { return true; } }
        protected override void custom_validate_opcode(Module ctx)
        {
            (geom != 0).AssertTrue();
            (ctype == b32).AssertTrue();
            (op == add || op == min || op == max || op == or || op == xor).AssertTrue();
            (op == add).AssertImplies(ctype == u32 || ctype == u64 || ctype == s32);
            (op == min || op == max).AssertImplies(ctype == u32 || ctype == s32);
            (op == and || op == or).AssertImplies(ctype == b32);
        }

        sured_p() { 1.UpTo(3).ForEach(_ => Operands.Add(null)); }
        public Expression a { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression b { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression c { get { return Operands[2]; } set { Operands[2] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(a, surfref).AssertTrue();
            if (geom == d1) (agree(b, s32) || agree(b, s32.v1)).AssertTrue();
            else if (geom == d2) agree(b, s32.v2).AssertTrue();
            else if (geom == d3) agree(b, s32.v4).AssertTrue();
            else throw AssertionHelper.Fail();
            agree(c, ctype).AssertTrue();
        }
    }
}