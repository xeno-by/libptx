using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Video
{
    [Ptxop20("vset.atype.btype.cmp d, a{.asel}, b{.bsel};")]
    [Ptxop20("vset.atype.btype.cmp.op2 d, a{.asel}, b{.bsel}, c;")]
    [Ptxop20("vset.atype.btype.cmp d.dsel, a{.asel}, b{.bsel}, c;")]
    internal class vset : ptxop
    {
        [Infix] public type atype { get; set; }
        [Infix] public type btype { get; set; }
        [Infix] public cmpop cmp { get; set; }
        [Infix] public op op2 { get; set; }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (atype == s32 || atype == u32).AssertTrue();
            (btype == s32 || btype == u32).AssertTrue();
            (cmp == eq || cmp == ne || cmp == lt || cmp == le || cmp == gt || cmp == ge).AssertTrue();
            (op2 == null || op2 == add || op2 == min || op2 == max).AssertTrue();
        }
    }
}