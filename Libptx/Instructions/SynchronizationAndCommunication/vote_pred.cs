using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop("vote.mode.pred d, {!}a;", SoftwareIsa.PTX_12)]
    public class vote_pred : ptxop
    {
        [Affix] public redm mode { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (mode != null).AssertTrue();
        }
    }
}