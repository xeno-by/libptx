using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;
using Libptx.Expressions;

namespace Libptx.Instructions.Video
{
    [Ptxop20("vshl.dtype.atype.u32{.sat}{.mode} d, a{.asel}, b{.bsel};")]
    [Ptxop20("vshl.dtype.atype.u32{.sat}{.mode}.op2 d, a{.asel}, b{.bsel}, c;")]
    [Ptxop20("vshl.dtype.atype.u32{.sat}{.mode} d.dsel, a{.asel}, b{.bsel}, c;")]
    public partial class vshl : ptxop
    {
        [Affix] public Type dtype { get; set; }
        [Affix] public Type atype { get; set; }
        [Affix] public Type btype { get; set; }
        [Affix] public bool sat { get; set; }
        [Affix] public vshm mode { get; set; }
        [Affix] public op op2 { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (dtype == s32 || dtype == u32).AssertTrue();
            (atype == s32 || atype == u32).AssertTrue();
            (btype == u32).AssertTrue();
            (mode == 0 || mode == vshm_clamp || mode == vshm_wrap).AssertTrue();
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
                agree(d, dtype, exact(sel)).AssertTrue();
                agree(a, atype, sel).AssertTrue();
                agree(b, btype, sel).AssertTrue();
                agree(c, dtype).AssertTrue();
            }
            else
            {
                agree(d, dtype).AssertTrue();
                agree(a, atype, sel).AssertTrue();
                agree(b, btype, sel).AssertTrue();
                agree_or_null(c, dtype).AssertTrue();
            }
        }
    }
}
