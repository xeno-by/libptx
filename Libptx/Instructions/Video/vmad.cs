using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Video
{
    [Ptxop20("vmad.dtype.atype.btype{.sat}{.scale} d, {-}a{.asel}, {-}b{.bsel}, {-}c;")]
    [Ptxop20("vmad.dtype.atype.btype.po{.sat}{.scale} d, a{.asel}, b{.bsel}, c;")]
    internal class vmad : ptxop
    {
        [Suffix] public type dtype { get; set; }
        [Suffix] public type atype { get; set; }
        [Suffix] public type btype { get; set; }
        [Suffix] public bool po { get; set; }
        [Suffix] public bool sat { get; set; }
        [Suffix] public scale scale { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
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