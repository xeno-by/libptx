using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Expressions.Immediate;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

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

        public bar_red_pred() { 1.UpTo(4).ForEach(_ => Operands.Add(null)); }
        public Expression p { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }
        public Expression c { get { return Operands[3]; } set { Operands[3] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            is_reg(p, type).AssertTrue();
            is_alu(a, u32).AssertTrue();
            is_alu_or_null(b, u32).AssertTrue();
            is_alu(c, pred, not).AssertTrue();

            var a_const = (a as Const).AssertNotNull();
            if (a_const != null)
            {
                var value = a_const.AssertCoerce<int>();
                (0 <= value && value <= 15).AssertTrue();
            }
        }
    }
}