using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    // todo. maybe split into bar_sync, bar_arrive and bar_red?

    [Ptxop("bar.sync a;")]
    [Ptxop20("bar.sync a, b;")]
    [Ptxop20("bar.arrive a, b;")]
    [Ptxop20("bar.red.popc.type d, a{, b}, {!}c;")]
    [Ptxop20("bar.red.op.pred p, a{, b}, {!}c;")]
    [DebuggerNonUserCode]
    internal class bar : ptxop
    {
        [Suffix] public bool sync { get; set; }
        [Suffix] public bool arrive { get; set; }
        [Suffix] public bool red { get; set; }
        [Suffix] public bool popc { get; set; }
        [Suffix] public type type { get; set; }
        [Suffix] public op op { get; set; }
        [Suffix] public bool pred { get; set; }

        protected override SoftwareIsa custom_swisa
        {
            get { return sync ? SoftwareIsa.PTX_10 : SoftwareIsa.PTX_20; }
        }

        protected override HardwareIsa custom_hwisa
        {
            get { return sync ? HardwareIsa.SM_10 : HardwareIsa.SM_20; }
        }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (sync ^ arrive).AssertTrue();
            (sync ^ red).AssertTrue();
            (arrive ^ red).AssertTrue();

            (popc ^ pred).AssertTrue();
            (popc == true).AssertImplies(type == u32);
            (popc == false).AssertImplies(type == null);
            (pred == true).AssertImplies(op == and || op == or);
            (pred == false).AssertImplies(op == null);

            // todo. implement this
            // Operands a, b, and d have type .u32; operands p and c are predicates.
            // Register operands, thread count ... introduced in PTX ISA version 2.0.
            // Register operands, thread count ... require sm_20 or later.
        }
    }
}