using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("bfind.type d, a;")]
    [Ptxop20("bfind.shiftamt.type d, a;")]
    [DebuggerNonUserCode]
    public partial class bfind : ptxop
    {
        [Affix] public bool shiftamt { get; set; }
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (type.is_int() && type.bits() >= 32).AssertTrue();
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