using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop20("testp.op.type p, a;")]
    [DebuggerNonUserCode]
    public partial class testp : ptxop
    {
        [Affix] public test op { get; set; }
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            type.isfloat().AssertTrue();
        }
    }
}