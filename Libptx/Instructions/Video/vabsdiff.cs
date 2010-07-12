using System;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Video
{
    [Ptxop20("vabsdiff.dtype.atype.btype{.sat} d, a{.asel}, b{.bsel};")]
    [Ptxop20("vabsdiff.dtype.atype.btype{.sat}.op2 d, a{.asel}, b{.bsel}, c;")]
    [Ptxop20("vabsdiff.dtype.atype.btype{.sat} d.dsel, a{.asel}, b{.bsel}, c;")]
    internal class vabsdiff : ptxop
    {
        [Infix] public type dtype { get; set; }
        [Infix] public type atype { get; set; }
        [Infix] public type btype { get; set; }
        [Infix] public bool sat { get; set; }
        [Infix] public op op2 { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (dtype == s32 || dtype == u32).AssertTrue();
            (atype == s32 || atype == u32).AssertTrue();
            (btype == s32 || btype == u32).AssertTrue();
            (op2 == null || op2 == add || op2 == min || op2 == max).AssertTrue();
        }
    }
}
