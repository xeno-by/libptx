using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop20("cvta.to.space.size  p, a;")]
    [DebuggerNonUserCode]
    public partial class cvta_to : ptxop
    {
        [Affix] public space space { get; set; }
        [Affix] public Type size { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (space == local || space == shared || space == global).AssertTrue();
            (size == u32 || size == u64).AssertTrue();
        }

        public Expression p { get; set; }
        public Expression a { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(p, size).AssertTrue();
            agree(a, size).AssertTrue();
        }
    }
}