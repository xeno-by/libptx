using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;

namespace Libptx.Instructions.LogicAndShift
{
    [Ptxop("shr.type d, a, b;")]
    [DebuggerNonUserCode]
    public partial class shr : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }

        protected override void custom_validate_opcode(Module ctx)
        {
            (type.is_bit() || type.is_int()).AssertTrue();
        }

        public Expression d { get; set; }
        public Expression a { get; set; }
        public Expression b { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, type).AssertTrue();
            agree(a, type).AssertTrue();
            agree(b, u32).AssertTrue();
        }
    }
}