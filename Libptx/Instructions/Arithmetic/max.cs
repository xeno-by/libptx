using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("max.type        d, a, b;")]
    [Ptxop("max{.ftz}.f32   d, a, b;")]
    [Ptxop("max.f64         d, a, b;")]
    [DebuggerNonUserCode]
    public partial class max : ptxop
    {
        [Affix] public bool ftz { get; set; }
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode()
        {
            (ftz == true).AssertImplies(type == f32);
        }

        public max() { 1.UpTo(3).ForEach(_ => Operands.Add(null)); }
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