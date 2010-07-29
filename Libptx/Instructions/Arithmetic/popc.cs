using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("popc.type d, a;")]
    [DebuggerNonUserCode]
    public partial class popc : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode()
        {
            (type.is_bit() && type.bits() >= 32).AssertTrue();
        }

        public popc() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands()
        {
            is_alu(d, u32).AssertTrue();
            is_alu(a, type).AssertTrue();
        }
    }
}