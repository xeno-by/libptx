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
    [Ptxop20("prmt.b32{.mode} d, a, b, c;")]
    [DebuggerNonUserCode]
    public partial class prmt : ptxop
    {
        [Affix] public Type type { get; set; }
        [Affix] public prmtm mode { get; set; }

        protected override void custom_validate_opcode()
        {
            (type == b32).AssertTrue();
        }

        public prmt() { 1.UpTo(4).ForEach(_ => Operands.Add(null)); }
        [Destination] public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }
        public Expression c { get { return Operands[3]; } set { Operands[3] = value; } }

        protected override void custom_validate_operands()
        {
            is_reg(d, type).AssertTrue();
            is_alu(a, type).AssertTrue();
            is_alu(b, type).AssertTrue();
            is_alu(c, type).AssertTrue();
        }
    }
}