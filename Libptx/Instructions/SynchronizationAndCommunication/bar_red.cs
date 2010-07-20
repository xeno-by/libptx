using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    // todo. maybe split into bar_sync, bar_arrive and bar_red?

    [Ptxop20("bar.red.popc.type d, a{, b}, {!}c;")]
    [Ptxop20("bar.red.op.pred p, a{, b}, {!}c;")]
    [DebuggerNonUserCode]
    public partial class bar_red : ptxop
    {
        [Affix] public bool popc { get; set; }
        [Affix] public Type type { get; set; }
        [Affix] public op op { get; set; }
        [Affix] public new bool pred { get; set; }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (popc ^ pred).AssertTrue();
            (popc == true).AssertImplies(type == u32);
            (popc == false).AssertImplies(type == null);
            (pred == true).AssertImplies(op == and || op == or);
            (pred == false).AssertImplies(op == 0);

            // todo. implement this
            // Operands a, b, and d have type .u32; operands p and c are predicates.
        }
    }
}