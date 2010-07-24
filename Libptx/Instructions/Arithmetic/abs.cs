using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;

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

        protected override void custom_validate_opcode(Module ctx)
        {
            (type.is_signed() || type.is_float()).AssertTrue();
            (ftz == true).AssertImplies(type == f32);
        }

        public Expression d { get; set; }
        public Expression a { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, type).AssertTrue();
            agree(a, type).AssertTrue();
        }
    }
}