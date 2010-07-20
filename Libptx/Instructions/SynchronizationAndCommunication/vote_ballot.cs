using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop("vote.ballot.type d, {!}a;", SoftwareIsa.PTX_20)]
    public partial class vote_ballot : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (type == b32).AssertTrue();
        }
    }
}