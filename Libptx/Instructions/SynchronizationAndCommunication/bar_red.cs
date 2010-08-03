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
    [Ptxop20("bar.red.op.type p, a, {!}c;")]
    [Ptxop20("bar.red.op.type p, a, b, {!}c;")]
    [DebuggerNonUserCode]
    public partial class bar_red : ptxop
    {
        [Affix] public op op { get; set; }
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode()
        {
            (op == and || op == or || op == popc).AssertTrue();
            (op == and || op == or).AssertImplies(type == pred);
            (op == popc).AssertImplies(type == u32);
        }

        public bar_red() { 1.UpTo(4).ForEach(_ => Operands.Add(null)); }
        [Destination] public Expression p { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }
        public Expression c { get { return Operands[3]; } set { Operands[3] = value; } }

        protected override void custom_validate_operands()
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