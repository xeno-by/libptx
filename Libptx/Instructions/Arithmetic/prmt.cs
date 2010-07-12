using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("prmt.b32{.mode} d, a, b, c;")]
    [DebuggerNonUserCode]
    internal class prmt : ptxop
    {
        [Infix] public type type { get; set; }
        [Infix] public prmtm mode { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (type == b32).AssertTrue();
        }
    }
}