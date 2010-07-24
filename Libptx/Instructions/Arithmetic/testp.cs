using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;
using Libptx.Expressions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("testp.op.type p, a;")]
    [DebuggerNonUserCode]
    public partial class testp : ptxop
    {
        [Affix] public test op { get; set; }
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            type.is_float().AssertTrue();
        }

        public Expression p { get; set; }
        public Expression a { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(p, pred).AssertTrue();
            agree(a, type).AssertTrue();
        }
    }
}