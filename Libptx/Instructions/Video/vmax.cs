using System;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Video
{
    [Ptxop20("vmax.dtype.atype.btype{.sat} d, a{.asel}, b{.bsel};")]
    [Ptxop20("vmax.dtype.atype.btype{.sat}.op2 d, a{.asel}, b{.bsel}, c;")]
    [Ptxop20("vmax.dtype.atype.btype{.sat} d.dsel, a{.asel}, b{.bsel}, c;")]
    internal class vmax : ptxop
    {
        [Affix] public type dtype { get; set; }
        [Affix] public type atype { get; set; }
        [Affix] public type btype { get; set; }
        [Affix] public bool sat { get; set; }
        [Affix] public op op2 { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (dtype == s32 || dtype == u32).AssertTrue();
            (atype == s32 || atype == u32).AssertTrue();
            (btype == s32 || btype == u32).AssertTrue();
            (op2 == null || op2 == add || op2 == min || op2 == max).AssertTrue();
        }
    }
}
