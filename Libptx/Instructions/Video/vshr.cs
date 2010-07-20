﻿using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Video
{
    [Ptxop20("vshr.dtype.atype.btype{.sat}{.mode} d, a, b;")]
    [Ptxop20("vshr.dtype.atype.btype{.sat}{.mode}.op2 d, a, b, c;")]
    [Ptxop20("vshr.dtype.atype.btype{.sat}{.mode} d.dsel, a, b, c;")]
    public class vshr : ptxop
    {
        [Affix] public Type dtype { get; set; }
        [Affix] public Type atype { get; set; }
        [Affix] public Type btype { get; set; }
        [Affix] public bool sat { get; set; }
        [Affix] public vshm mode { get; set; }
        [Affix] public op op2 { get; set; }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (dtype == s32 || dtype == u32).AssertTrue();
            (atype == s32 || atype == u32).AssertTrue();
            (btype == u32).AssertTrue();
            (mode == null || mode == vshm_clamp || mode == vshm_wrap).AssertTrue();
            (op2 == null || op2 == add || op2 == min || op2 == max).AssertTrue();
        }
    }
}
