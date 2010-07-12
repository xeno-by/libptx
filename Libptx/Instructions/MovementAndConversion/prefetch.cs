using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
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
        [Mod] public bool u { get; set; }
        [Infix] public ss space { get; set; }
        [Infix] public cachelevel level { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (u == true).AssertEquiv(space == null && level == L1);
            (space == local || space == global).AssertTrue();
        }
    }
}