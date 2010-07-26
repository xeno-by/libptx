using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("bfe.type d, a, b, c;")]
    [DebuggerNonUserCode]
    public partial class bfe : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (type.is_int() && type.bits() >= 32).AssertTrue();
        }

        public bfe() { 1.UpTo(4).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }
        public Expression c { get { return Operands[3]; } set { Operands[3] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, type).AssertTrue();
            agree(a, type).AssertTrue();
            agree(b, u32).AssertTrue();
            agree(c, u32).AssertTrue();
        }
    }
}