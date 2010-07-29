using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop("cvt{.irnd}{.ftz}{.sat}.dtype.atype d, a;")]
    [Ptxop("cvt{.frnd}{.ftz}{.sat}.dtype.atype d, a;")]
    [DebuggerNonUserCode]
    public partial class cvt : ptxop
    {
        [Affix] public irnd irnd { get; set; }
        [Affix] public frnd frnd { get; set; }
        [Affix] public bool ftz { get; set; }
        [Affix] public bool sat { get; set; }
        [Affix] public Type dtype { get; set; }
        [Affix] public Type atype { get; set; }

        protected override bool allow_int8 { get { return true; } }
        protected override bool allow_float16 { get { return true; } }
        protected override void custom_validate_opcode()
        {
            var i2i = atype.is_int() && dtype.is_int();
            var i2f = atype.is_int() && dtype.is_float();
            var f2i = atype.is_float() && dtype.is_int();
            var f2f = atype.is_float() && dtype.is_float();
            (i2i || i2f || f2i || f2f).AssertTrue();

            (irnd != 0).AssertImplies(f2i || f2f);
            (frnd != 0).AssertImplies(i2f || f2f);
            (irnd != 0 && frnd != 0).AssertFalse();
        }

        public cvt() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands()
        {
            is_relaxed_reg(d, dtype).AssertTrue();
            is_relaxed_alu_or_sreg(a, atype).AssertTrue();
        }
    }
}
