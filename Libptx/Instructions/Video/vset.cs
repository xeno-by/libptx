using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;

namespace Libptx.Instructions.Video
{
    [Ptxop20("vset.atype.btype.cmp d, a{.asel}, b{.bsel};")]
    [Ptxop20("vset.atype.btype.cmp.op2 d, a{.asel}, b{.bsel}, c;")]
    [Ptxop20("vset.atype.btype.cmp d.dsel, a{.asel}, b{.bsel}, c;")]
    public partial class vset : ptxop
    {
        [Affix] public Type atype { get; set; }
        [Affix] public Type btype { get; set; }
        [Affix] public cmp cmp { get; set; }
        [Affix] public op op2 { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (atype == s32 || atype == u32).AssertTrue();
            (btype == s32 || btype == u32).AssertTrue();
            (cmp == eq || cmp == ne || cmp == lt || cmp == le || cmp == gt || cmp == ge).AssertTrue();
            (op2 == 0 || op2 == add || op2 == min || op2 == max).AssertTrue();
        }

        public Expression d { get; set; }
        public Expression a { get; set; }
        public Expression b { get; set; }
        public Expression c { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            var datamerge = op2 == 0 && c != null;
            if (datamerge)
            {
                agree(d, u32, exact(sel)).AssertTrue();
                agree(a, atype, sel).AssertTrue();
                agree(b, btype, sel).AssertTrue();
                agree(c, u32).AssertTrue();
            }
            else
            {
                agree(d, u32).AssertTrue();
                agree(a, atype, sel).AssertTrue();
                agree(b, btype, sel).AssertTrue();
                agree_or_null(c, u32).AssertTrue();
            }
        }
    }
}