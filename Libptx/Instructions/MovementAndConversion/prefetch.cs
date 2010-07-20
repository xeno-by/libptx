using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop20("prefetch{.space}.level  [a];")]
    [Ptxop20("prefetchu.L1            [a];")]
    [DebuggerNonUserCode]
    public partial class prefetch : ptxop
    {
        [Mod] public bool u { get; set; }
        [Affix] public space space { get; set; }
        [Affix] public cachelevel level { get; set; }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (u == true).AssertEquiv(space == 0 && level == L1);
            (space == local || space == global).AssertTrue();
        }
    }
}