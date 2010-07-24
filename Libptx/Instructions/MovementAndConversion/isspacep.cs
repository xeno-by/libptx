using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop20("isspacep.space p, a;")]
    [DebuggerNonUserCode]
    public partial class isspacep : ptxop
    {
        [Affix] public space space { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (space == local || space == shared || space == global).AssertTrue();
        }

        public Expression p { get; set; }
        public Expression a { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(p, pred).AssertTrue();
            (agree(a, u32) || agree(a, u64)).AssertTrue();
        }
    }
}