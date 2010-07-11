using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop20("isspacep.space p, a;")]
    [DebuggerNonUserCode]
    internal class isspacep : ptxop
    {
        [Suffix] public ss space { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (space == local || space == shared || space == global).AssertTrue();
        }
    }
}