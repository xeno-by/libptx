using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quantas;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("testp.op.type p, a;")]
    [DebuggerNonUserCode]
    internal class testp : ptxop
    {
        [Affix] public testpop op { get; set; }
        [Affix] public type type { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            type.isfloat().AssertTrue();
        }
    }
}