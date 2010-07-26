using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;
using Type = Libptx.Common.Types.Type;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop20("sust.p.geom{.cop}.ctype.clampm    [a, b], c;")]
    [DebuggerNonUserCode]
    public partial class sust_p : ptxop
    {
        [Affix] public geom geom { get; set; }
        [Affix] public cop cop { get; set; }
        [Affix] public Type ctype { get; set; }
        [Affix] public clampm clampm { get; set; }

        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_vec { get { return true; } }
        protected override void custom_validate_opcode(Module ctx)
        {
            (geom != 0).AssertTrue();
            (cop == 0 || cop == wb || cop == cg || cop == cs || cop == wt).AssertTrue();
            (ctype.is32() && (ctype.is_scalar() || ctype.is_v2() || ctype.is_v4())).AssertTrue();
        }

        public sust_p() { 1.UpTo(3).ForEach(_ => Operands.Add(null)); }
        public Expression a { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression b { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression c { get { return Operands[2]; } set { Operands[2] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            is_alu(a, surfref).AssertTrue();
            if (geom == d1) (is_alu(b, s32) || is_alu(b, s32.v1)).AssertTrue();
            else if (geom == d2) is_alu(b, s32.v2).AssertTrue();
            else if (geom == d3) is_alu(b, s32.v4).AssertTrue();
            else throw AssertionHelper.Fail();
            is_alu(c, ctype).AssertTrue();
        }
    }
}