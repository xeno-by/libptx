using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Video
{
    [Ptxop20("vshr.dtype.atype.btype{.sat}{.mode} d, a{.asel}, b{.bsel};")]
    [Ptxop20("vshr.dtype.atype.btype{.sat}{.mode}.op2 d, a{.asel}, b{.bsel}, c;")]
    [Ptxop20("vshr.dtype.atype.btype{.sat}{.mode} d.dsel, a{.asel}, b{.bsel}, c;")]
    internal class vshr : ptxop
    {
        [Suffix] public type dtype { get; set; }
        [Suffix] public type atype { get; set; }
        [Suffix] public type btype { get; set; }
        [Suffix] public bool sat { get; set; }
        [Suffix] public vshm mode { get; set; }
        [Suffix] public op op2 { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (dtype == s32 || dtype == u32).AssertTrue();
            (atype == s32 || atype == u32).AssertTrue();
            (btype == u32).AssertTrue();
            (mode == null || mode == vshm_clamp || mode == vshm_wrap).AssertTrue();
            (op2 == null || op2 == add || op2 == min || op2 == max).AssertTrue();
        }
    }
}
