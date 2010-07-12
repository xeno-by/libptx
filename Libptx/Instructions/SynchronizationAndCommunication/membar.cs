using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop10("membar.level;")]
    internal class membar : ptxop
    {
        [Infix] public barlevel level { get; set; }

        protected override SoftwareIsa custom_swisa
        {
            get { return level == sys ? SoftwareIsa.PTX_20 : SoftwareIsa.PTX_10; }
        }

        protected override HardwareIsa custom_hwisa
        {
            get { return level == sys ? HardwareIsa.SM_20 : HardwareIsa.SM_10; }
        }
    }
}