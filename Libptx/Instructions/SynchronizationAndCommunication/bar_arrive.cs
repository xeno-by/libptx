using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop20("bar.arrive a, b;")]
    [DebuggerNonUserCode]
    public partial class bar_arrive : ptxop
    {
        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            // todo. implement the following:
            // Operands a, b, and d have type .u32; operands p and c are predicates.
        }
    }
}