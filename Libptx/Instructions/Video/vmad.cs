using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Video
{
    [Ptxop20("vmad.dtype.atype.btype{.sat}{.scale} d, {-}a, {-}b, {-}c;")]
    [Ptxop20("vmad.dtype.atype.btype.po{.sat}{.scale} d, a, b, c;")]
    public class vmad : ptxop
    {
        [Affix] public Type dtype { get; set; }
        [Affix] public Type atype { get; set; }
        [Affix] public Type btype { get; set; }
        [Affix] public bool po { get; set; }
        [Affix] public bool sat { get; set; }
        [Affix] public scale scale { get; set; }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (dtype == s32 || dtype == u32).AssertTrue();
            (atype == s32 || atype == u32).AssertTrue();
            (btype == s32 || btype == u32).AssertTrue();

            // todo. implement this:
            // Source operands may not be negated in .po mode.
            // Depending on the sign of the a and b operands, and the operand negates, the following combinations of operands are supported for VMAD:
        }
    }
}