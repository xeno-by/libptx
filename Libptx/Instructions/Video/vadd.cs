using Libptx.Common.Types;
using Libptx.Common.Annotations.Quanta;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.Video
{
    [Ptxop20("vadd.dtype.atype.btype{.sat} d, a{.asel}, b{.bsel};")]
    [Ptxop20("vadd.dtype.atype.btype{.sat}.op2 d, a{.asel}, b{.bsel}, c;")]
    [Ptxop20("vadd.dtype.atype.btype{.sat} d.dsel, a{.asel}, b{.bsel}, c;")]
    public partial class vadd : ptxop
    {
        [Affix] public Type dtype { get; set; }
        [Affix] public Type atype { get; set; }
        [Affix] public Type btype { get; set; }
        [Affix] public bool sat { get; set; }
        [Affix] public op op2 { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (dtype == s32 || dtype == u32).AssertTrue();
            (atype == s32 || atype == u32).AssertTrue();
            (btype == s32 || btype == u32).AssertTrue();
            (op2 == 0 || op2 == add || op2 == min || op2 == max).AssertTrue();
        }

        vadd() { 1.UpTo(4).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }
        public Expression c { get { return Operands[3]; } set { Operands[3] = value; } }

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
