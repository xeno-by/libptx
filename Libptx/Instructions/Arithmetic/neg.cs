using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Edsl.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("neg.type        d, a;")]
    [Ptxop("neg{.ftz}.f32   d, a;")]
    [Ptxop("neg.f64         d, a;")]
    [DebuggerNonUserCode]
    public class neg : ptxop
    {
        [Affix] public bool ftz { get; set; }
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (ftz == true).AssertImplies(type == f32);
            (type.issigned() || type.isfloat()).AssertTrue();
        }
    }
}