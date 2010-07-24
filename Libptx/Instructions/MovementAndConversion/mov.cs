using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Expressions;
using XenoGears.Assertions;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop("mov.type d, a;")]
    [DebuggerNonUserCode]
    public partial class mov : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override bool allow_pred { get { return true; } }
        protected override bool allow_vec { get { return true; } }

        public Expression d { get; set; }
        public Expression a { get; set; }

        protected override bool allow_ptr { get { return true; } }
        protected override bool allow_special { get { return true; } }
        protected override void custom_validate_operands(Module ctx)
        {
            (agree(d, type) && is_reg(d)).AssertTrue();
            agree(a, type).AssertTrue();
        }
    }
}