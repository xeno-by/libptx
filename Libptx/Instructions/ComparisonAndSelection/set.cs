using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;

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
        protected override void custom_validate_opcode(Module ctx)
        {
            (boolop == 0 || boolop == and || boolop == or || boolop == xor).AssertTrue();
            (dtype == u32 || dtype == s32 || dtype == f32).AssertTrue();
        }

        public Expression d { get; set; }
        public Expression a { get; set; }
        public Expression b { get; set; }
        public Expression c { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, dtype).AssertTrue();
            agree(a, stype).AssertTrue();
            agree(b, stype).AssertTrue();
            agree_or_null(c, pred, not).AssertTrue();
        }
    }
}