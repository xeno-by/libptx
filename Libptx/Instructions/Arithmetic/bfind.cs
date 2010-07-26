using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("bfind.type d, a;")]
    [Ptxop20("bfind.shiftamt.type d, a;")]
    [DebuggerNonUserCode]
    public partial class bfind : ptxop
    {
        [Affix] public bool shiftamt { get; set; }
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (type.is_int() && type.bits() >= 32).AssertTrue();
        }

        public bfind() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, u32).AssertTrue();
            agree(a, type).AssertTrue();
        }
    }
}