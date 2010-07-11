using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop20("prefetch{.space}.level  [a];")]
    [Ptxop20("prefetchu.L1            [a];")]
    [DebuggerNonUserCode]
    internal class prefetch : ptxop
    {
        [Endian] public bool u { get; set; }
        [Suffix] public ss space { get; set; }
        [Suffix] public cachelevel level { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (u == true).AssertEquiv(space == null && level == L1);
            (space == local || space == global).AssertTrue();
        }
    }
}