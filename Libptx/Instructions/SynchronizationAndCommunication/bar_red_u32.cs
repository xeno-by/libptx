using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop20("bar.red.popc.type d, a{, b}, {!}c;")]
    [DebuggerNonUserCode]
    public partial class bar_red_u32 : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (type == u32).AssertTrue();
        }

        public Expression d { get; set; }
        public Expression a { get; set; }
        public Expression b { get; set; }
        public Expression c { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, type).AssertTrue();
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