using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop10("vote.mode.pred d, {!}a;", SoftwareIsa.PTX_12)]
    internal class vote_pred : ptxop
    {
        [Infix] public redm mode { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (mode != null).AssertTrue();
        }
    }
}