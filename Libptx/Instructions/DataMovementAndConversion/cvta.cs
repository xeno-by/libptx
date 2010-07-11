using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.DataMovementAndConversion
{
    [Ptxop20("cvta.space.size     p, a;")]
    [Ptxop20("cvta.space.size     p, var;")]
    [Ptxop20("cvta.to.space.size  p, a;")]
    [DebuggerNonUserCode]
    internal class cvta : ptxop
    {
        [Suffix] public bool to { get; set; }
        [Suffix] public ss space { get; set; }
        [Suffix] public size size { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (space == local || space == shared || space == global).AssertTrue();
            (size != null).AssertTrue();
        }
    }
}