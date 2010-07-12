using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("testp.op.type p, a;")]
    [DebuggerNonUserCode]
    internal class testp : ptxop
    {
        [Infix] public testpop op { get; set; }
        [Infix] public type type { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            type.isfloat().AssertTrue();
        }
    }
}