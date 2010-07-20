using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop("vote.ballot.type d, {!}a;", SoftwareIsa.PTX_20)]
    public class vote_ballot : ptxop
    {
        [Affix] public type type { get; set; }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (type == b32).AssertTrue();
        }
    }
}