using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop20("vote.ballot.b32 d, {!}a;")]
    public partial class vote_ballot : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (type == b32).AssertTrue();
        }

        public Expression d { get; set; }
        public Expression a { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, type).AssertTrue();
            agree(a, pred, not).AssertTrue();
        }
    }
}