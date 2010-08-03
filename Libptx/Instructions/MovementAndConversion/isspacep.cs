using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop20("isspacep.space p, a;")]
    [DebuggerNonUserCode]
    public partial class isspacep : ptxop
    {
        [Affix] public space space { get; set; }

        protected override void custom_validate_opcode()
        {
            (space == local || space == shared || space == global).AssertTrue();
        }

        public isspacep() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        [Destination] public Expression p { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands()
        {
            is_reg(p, pred).AssertTrue();
            (is_alu(a, u32) || is_alu(a, u64)).AssertTrue();
        }
    }
}