using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;
using Libptx.Expressions;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop20("bar.red.op.pred p, a{, b}, {!}c;")]
    [DebuggerNonUserCode]
    public partial class bar_red_pred : ptxop
    {
        [Affix] public Type type { get; set; }
        [Affix] public op op { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (type == pred).AssertTrue();
            (op == and || op == or).AssertTrue();
        }

        public Expression p { get; set; }
        public Expression a { get; set; }
        public Expression b { get; set; }
        public Expression c { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(p, type).AssertTrue();
            agree(a, u32).AssertTrue();
            agree_or_null(b, u32).AssertTrue();
            agree(c, pred, not).AssertTrue();

            var a_const = (a as Const).AssertNotNull();
            if (a_const != null)
            {
                var value = @const.AssertCoerce<int>();
                (0 <= value && value <= 15).AssertTrue();
            }
        }
    }
}