using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;

namespace Libptx.Instructions.LogicAndShift
{
    [Ptxop("cnot.type d, a;")]
    [DebuggerNonUserCode]
    public partial class cnot : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            type.isbit().AssertTrue();
        }
    }
}