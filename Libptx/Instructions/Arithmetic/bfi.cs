using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("bfi.type f, a, b, c, d;")]
    [DebuggerNonUserCode]
    public partial class bfi : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (type.is_bit() && type.bits() >= 32).AssertTrue();
        }

        public Expression f { get; set; }
        public Expression a { get; set; }
        public Expression b { get; set; }
        public Expression c { get; set; }
        public Expression d { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(f, type).AssertTrue();
            agree(a, type).AssertTrue();
            agree(b, type).AssertTrue();
            agree(c, u32).AssertTrue();
            agree(d, u32).AssertTrue();
        }
    }
}