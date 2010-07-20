using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;

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

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (type.issigned() || type.isfloat()).AssertTrue();
            (ftz == true).AssertImplies(type == f32);
        }
    }
}