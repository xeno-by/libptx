using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("bfind.type d, a;")]
    [Ptxop20("bfind.shiftamt.type d, a;")]
    [DebuggerNonUserCode]
    internal class bfind : ptxop
    {
        [Infix] public bool shiftamt { get; set; }
        [Infix] public type type { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (type.isint() && type.bits() >= 32).AssertTrue();
        }
    }
}