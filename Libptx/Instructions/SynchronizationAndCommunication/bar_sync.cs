using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop10("bar.sync a;")]
    [Ptxop20("bar.sync a, b;")]
    [DebuggerNonUserCode]
    internal class bar_sync : ptxop
    {
        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            // todo. implement the following:
            // Operands a, b, and d have type .u32; operands p and c are predicates.
            // Register operands, thread count ... introduced in PTX ISA version 2.0.
            // Register operands, thread count ... require sm_20 or later.
        }
    }
}