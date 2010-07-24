using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using Type=Libptx.Common.Types.Type;

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
        }

        public Expression p { get; set; }
        public Expression a { get; set; }
        public Expression b { get; set; }
        public Expression c { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(p, pred, couple).AssertTrue();
            agree(a, type).AssertTrue();
            agree(b, type).AssertTrue();
            agree_or_null(c, pred, not).AssertTrue();
        }
    }
}