using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

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

        public cvta_to() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        public Expression p { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            is_reg(p, size).AssertTrue();
            is_alu(a, size).AssertTrue();
        }
    }
}