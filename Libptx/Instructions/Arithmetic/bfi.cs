using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("bfi.type f, a, b, c, d;")]
    [DebuggerNonUserCode]
    public partial class bfi : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (type.is_bit() && type.bits() >= 32).AssertTrue();
        }

        public bfi() { 1.UpTo(5).ForEach(_ => Operands.Add(null)); }
        public Expression f { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }
        public Expression c { get { return Operands[3]; } set { Operands[3] = value; } }
        public Expression d { get { return Operands[4]; } set { Operands[4] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            is_alu(f, type).AssertTrue();
            is_alu(a, type).AssertTrue();
            is_alu(b, type).AssertTrue();
            is_alu(c, u32).AssertTrue();
            is_alu(d, u32).AssertTrue();
        }
    }
}