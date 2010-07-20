using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Video
{
    [Ptxop20("vset.atype.btype.cmp d, a, b;")]
    [Ptxop20("vset.atype.btype.cmp.op2 d, a, b, c;")]
    [Ptxop20("vset.atype.btype.cmp d.dsel, a, b, c;")]
    public class vset : ptxop
    {
        [Affix] public Type atype { get; set; }
        [Affix] public Type btype { get; set; }
        [Affix] public cmp cmp { get; set; }
        [Affix] public op op2 { get; set; }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (atype == s32 || atype == u32).AssertTrue();
            (btype == s32 || btype == u32).AssertTrue();
            (cmp == eq || cmp == ne || cmp == lt || cmp == le || cmp == gt || cmp == ge).AssertTrue();
            (op2 == 0 || op2 == add || op2 == min || op2 == max).AssertTrue();
        }
    }
}