using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;
using Type = Libptx.Common.Types.Type;
using Libptx.Expressions;

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

        protected override bool allow_bit8 { get { return true; } }
        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override void custom_validate_opcode(Module ctx)
        {
            (geom != 0).AssertTrue();
            (cop == 0 || cop == wb || cop == cg || cop == cs || cop == wt).AssertTrue();
            (ctype.is32() && ctype.is_v4()).AssertTrue();
        }

        public Expression a { get; set; }
        public Expression b { get; set; }
        public Expression c { get; set; }

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