using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Instructions.Annotations;
using Libptx.Expressions;
using XenoGears.Assertions;
using Type=Libptx.Common.Types.Type;

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
            (ctype == s32 || ctype == f32).AssertTrue();
        }

        public Expression d { get; set; }
        public Expression a { get; set; }
        public Expression b { get; set; }
        public Expression c { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, dtype).AssertTrue();
            agree(a, dtype).AssertTrue();
            agree(b, dtype).AssertTrue();
            agree(c, ctype).AssertTrue();
        }
    }
}