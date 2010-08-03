using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("abs.type        d, a;")]
    [Ptxop("abs{.ftz}.f32   d, a;")]
    [Ptxop("abs.f64         d, a;")]
    [DebuggerNonUserCode]
    public partial class abs : ptxop
    {
        [Affix] public bool ftz { get; set; }
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode()
        {
            (type.is_signed() || type.is_float()).AssertTrue();
            (ftz == true).AssertImplies(type == f32);
        }

        public abs() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        [Destination] public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands()
        {
            is_reg(d, type).AssertTrue();
            is_alu(a, type).AssertTrue();
        }
    }
}