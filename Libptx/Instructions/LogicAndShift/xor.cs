using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;

namespace Libptx.Instructions.LogicAndShift
{
    [Ptxop("xor.type d, a, b;")]
    [DebuggerNonUserCode]
    public partial class xor : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override bool allow_pred { get { return true; } }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (type.isbit() || type.ispred()).AssertTrue();
        }
    }
}