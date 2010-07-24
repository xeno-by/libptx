using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Expressions;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Instructions.ComparisonAndSelection
{
    [Ptxop("selp.type d, a, b, c;")]
    [DebuggerNonUserCode]
    public partial class selp : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }

        public Expression d { get; set; }
        public Expression a { get; set; }
        public Expression b { get; set; }
        public Expression c { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, type).AssertTrue();
            agree(a, type).AssertTrue();
            agree(b, type).AssertTrue();
            agree(c, pred).AssertTrue();
        }
    }
}