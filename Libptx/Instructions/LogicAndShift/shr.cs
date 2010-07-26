using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.LogicAndShift
{
    [Ptxop("shr.type d, a, b;")]
    [DebuggerNonUserCode]
    public partial class shr : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }

        protected override void custom_validate_opcode(Module ctx)
        {
            (type.is_bit() || type.is_int()).AssertTrue();
        }

        public shr() { 1.UpTo(3).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            is_reg(d, type).AssertTrue();
            is_alu(a, type).AssertTrue();
            is_alu(b, u32).AssertTrue();
        }
    }
}