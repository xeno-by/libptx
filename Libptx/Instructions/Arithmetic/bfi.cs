using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("bfi.type f, a, b, c, d;")]
    [DebuggerNonUserCode]
    internal class bfi : ptxop
    {
        [Suffix] public type type { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (type.isbit() && type.bits() >= 32).AssertTrue();
        }
    }
}