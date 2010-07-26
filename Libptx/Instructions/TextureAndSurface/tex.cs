using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using Libptx.Expressions;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop("tex.geom.dtype.btype d, [a, c];")]
    [Ptxop("tex.geom.dtype.btype d, [a, b, c];")]
    [DebuggerNonUserCode]
    public partial class tex : ptxop
    {
        [Affix] public geom geom { get; set; }
        [Affix] public Type dtype { get; set; }
        [Affix] public Type btype { get; set; }

        protected override bool allow_vec { get { return true; } }
        protected override void custom_validate_opcode(Module ctx)
        {
            (geom != 0).AssertTrue();
            (dtype.is32() && dtype.is_v4()).AssertTrue();
            (btype == s32 || btype == f32).AssertTrue();
        }

        public tex() { 1.UpTo(4).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }
        public Expression c { get { return Operands[3]; } set { Operands[3] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            is_alu(d, dtype).AssertTrue();
            is_alu(a, texref).AssertTrue();
            is_alu_or_null(b, samplerref).AssertTrue();
            (b != null).AssertImplies(!ctx.UnifiedTexturing);
            if (geom == d1) (is_alu(c, btype) || is_alu(c, btype.v1) || is_alu(c, btype.v4)).AssertTrue();
            else if (geom == d2) (is_alu(c, btype.v2) || is_alu(c, btype.v4)).AssertTrue();
            else if (geom == d3) is_alu(c, btype.v4).AssertTrue();
            else throw AssertionHelper.Fail();
        }
    }
}