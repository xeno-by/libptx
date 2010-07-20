using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("prmt.b32{.mode} d, a, b, c;")]
    [DebuggerNonUserCode]
    public partial class prmt : ptxop
    {
        [Affix] public Type type { get; set; }
        [Affix] public prmtm mode { get; set; }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (type == b32).AssertTrue();
        }
    }
}