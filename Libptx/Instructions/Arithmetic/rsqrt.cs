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
    [Ptxop("rsqrt.approx{.ftz}.f32  d, a;")]
    [Ptxop("rsqrt.approx.f64        d, a;")]
    [DebuggerNonUserCode]
    public partial class rsqrt : ptxop
    {
        [Affix(SoftwareIsa.PTX_14)] public bool approx { get; set; }
        [Affix(SoftwareIsa.PTX_14)] public bool ftz { get; set; }
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            approx.AssertTrue();
            (ftz == true).AssertImplies(type == f32);
            type.is_float().AssertTrue();

            (ctx.Version >= SoftwareIsa.PTX_14 && type == f64).AssertImplies(approx);
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