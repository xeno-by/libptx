using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;

namespace Libptx.Instructions.LogicAndShift
{
    [Ptxop("not.type d, a;")]
    [DebuggerNonUserCode]
    public partial class not : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override bool allow_pred { get { return true; } }

        protected override void custom_validate_opcode(Module ctx)
        {
            (type.is_bit() || type.is_pred()).AssertTrue();
        }

        public Expression d { get; set; }
        public Expression a { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, type).AssertTrue();
            agree(a, type).AssertTrue();
        }
    }
}