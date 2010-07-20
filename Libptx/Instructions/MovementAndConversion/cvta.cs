using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop20("cvta.space.size     p, a;")]
    [Ptxop20("cvta.space.size     p, var;")]
    [Ptxop20("cvta.to.space.size  p, a;")]
    [DebuggerNonUserCode]
    public partial class cvta : ptxop
    {
        [Affix] public bool to { get; set; }
        [Affix] public space space { get; set; }
        [Affix] public size size { get; set; }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (space == local || space == shared || space == global).AssertTrue();
            (size != 0).AssertTrue();
        }
    }
}