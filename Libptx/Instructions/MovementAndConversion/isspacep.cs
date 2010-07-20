using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop20("isspacep.space p, a;")]
    [DebuggerNonUserCode]
    public class isspacep : ptxop
    {
        [Affix] public space space { get; set; }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (space == local || space == shared || space == global).AssertTrue();
        }
    }
}