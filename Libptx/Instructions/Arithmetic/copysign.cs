using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("copysign.type d, a, b;")]
    [DebuggerNonUserCode]
    public partial class copysign : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode()
        {
            type.is_float().AssertTrue();
        }

        public copysign() { 1.UpTo(3).ForEach(_ => Operands.Add(null)); }
        [Destination] public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }

        protected override void custom_validate_operands()
        {
            is_reg(d, type).AssertTrue();
            is_alu(a, type).AssertTrue();
            is_alu(b, type).AssertTrue();
        }
    }
}