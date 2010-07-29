using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("testp.op.type p, a;")]
    [DebuggerNonUserCode]
    public partial class testp : ptxop
    {
        [Affix] public test op { get; set; }
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode()
        {
            type.is_float().AssertTrue();
        }

        public testp() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        public Expression p { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands()
        {
            is_reg(p, pred).AssertTrue();
            is_alu(a, type).AssertTrue();
        }
    }
}