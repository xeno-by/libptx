using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Expressions;
using XenoGears.Assertions;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop20("bar.arrive a, b;")]
    [DebuggerNonUserCode]
    public partial class bar_arrive : ptxop
    {
        protected override void custom_validate_opcode(Module ctx)
        {
            // todo. implement the following:
            // Operands a, b, and d have type .u32; operands p and c are predicates.
        }

        public Expression a { get; set; }
        public Expression b { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(a, u32).AssertTrue();
            agree(b, u32).AssertTrue();

            var a_const = (a as Const).AssertNotNull();
            if (a_const != null)
            {
                var value = @const.AssertCoerce<int>();
                (0 <= value && value <= 15).AssertTrue();
            }
        }
    }
}