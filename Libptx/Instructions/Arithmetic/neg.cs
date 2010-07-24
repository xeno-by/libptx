using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("neg.type        d, a;")]
    [Ptxop("neg{.ftz}.f32   d, a;")]
    [Ptxop("neg.f64         d, a;")]
    [DebuggerNonUserCode]
    public partial class neg : ptxop
    {
        [Affix] public bool ftz { get; set; }
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (ftz == true).AssertImplies(type == f32);
            (type.is_signed() || type.is_float()).AssertTrue();
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