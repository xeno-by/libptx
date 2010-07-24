using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("popc.type d, a;")]
    [DebuggerNonUserCode]
    public partial class popc : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (type.is_bit() && type.bits() >= 32).AssertTrue();
        }

        public Expression d { get; set; }
        public Expression a { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, u32).AssertTrue();
            agree(a, type).AssertTrue();
        }
    }
}