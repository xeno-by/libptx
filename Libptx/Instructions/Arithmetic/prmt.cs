using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;
using Libptx.Expressions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("prmt.b32{.mode} d, a, b, c;")]
    [DebuggerNonUserCode]
    public partial class prmt : ptxop
    {
        [Affix] public Type type { get; set; }
        [Affix] public prmtm mode { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (type == b32).AssertTrue();
        }

        public Expression d { get; set; }
        public Expression a { get; set; }
        public Expression b { get; set; }
        public Expression c { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, type).AssertTrue();
            agree(a, type).AssertTrue();
            agree(b, type).AssertTrue();
            agree(c, type).AssertTrue();
        }
    }
}