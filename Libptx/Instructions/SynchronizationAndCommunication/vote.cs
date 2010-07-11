using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    // todo. maybe split into vote_pred and vote_ballot?

    [Ptxop("vote.mode.pred d, {!}a;", SoftwareIsa.PTX_12)]
    [Ptxop("vote.ballot.type d, {!}a;", SoftwareIsa.PTX_20)]
    internal class vote : ptxop
    {
        [Suffix] public redm mode { get; set; }
        [Suffix] public bool pred { get; set; }
        [Suffix] public bool ballot { get; set; }
        [Suffix] public type type { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (pred || ballot).AssertTrue();
            (pred == true).AssertImplies(mode != null);
            (pred == false).AssertImplies(mode == null);
            (ballot == true).AssertImplies(type == b32);
            (ballot == false).AssertImplies(type == null);
        }
    }
}