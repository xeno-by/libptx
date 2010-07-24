using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using Libptx.Expressions;
using XenoGears.Assertions;

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

        protected override void custom_validate_opcode(Module ctx)
        {
            (geom != 0).AssertTrue();
            (dtype.is32() && dtype.is_v4()).AssertTrue();
            (btype == s32 || btype == f32).AssertTrue();
        }

        public Expression d { get; set; }
        public Expression a { get; set; }
        public Expression c { get; set; }
        public Expression b { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, dtype).AssertTrue();
            agree(a, texref).AssertTrue();
            agree_or_null(b, samplerref).AssertTrue();
            (b != null).AssertImplies(!ctx.UnifiedTexturing);
            if (geom == d1) (agree(c, btype) || agree(c, btype.v1) || agree(c, btype.v4)).AssertTrue();
            else if (geom == d2) (agree(c, btype.v2) || agree(c, btype.v4)).AssertTrue();
            else if (geom == d3) agree(c, btype.v4).AssertTrue();
            else throw AssertionHelper.Fail();
        }
    }
}