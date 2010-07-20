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
    [Ptxop20("bfind.type d, a;")]
    [Ptxop20("bfind.shiftamt.type d, a;")]
    [DebuggerNonUserCode]
    public class bfind : ptxop
    {
        [Affix] public bool shiftamt { get; set; }
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (type.isint() && type.bits() >= 32).AssertTrue();
        }
    }
}