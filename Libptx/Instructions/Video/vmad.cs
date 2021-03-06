using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.Video
{
    [Ptxop20("vmad.dtype.atype.btype{.sat}{.scale} d, {-}a{.asel}, {-}b{.asel}, {-}c;")]
    [Ptxop20("vmad.dtype.atype.btype.po{.sat}{.scale} d, a{.asel}, b{.asel}, c;")]
    [DebuggerNonUserCode]
    public partial class vmad : ptxop
    {
        [Affix] public Type dtype { get; set; }
        [Affix] public Type atype { get; set; }
        [Affix] public Type btype { get; set; }
        [Affix] public bool po { get; set; }
        [Affix] public bool sat { get; set; }
        [Affix] public scale scale { get; set; }

        protected override void custom_validate_opcode()
        {
            (dtype == s32 || dtype == u32).AssertTrue();
            (atype == s32 || atype == u32).AssertTrue();
            (btype == s32 || btype == u32).AssertTrue();
        }

        public vmad() { 1.UpTo(4).ForEach(_ => Operands.Add(null)); }
        [Destination] public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }
        public Expression c { get { return Operands[3]; } set { Operands[3] = value; } }

        protected override void custom_validate_operands()
        {
            // todo. implement this:
            //
            // Depending on the sign of the a and b operands, and the operand negates,
            // the following combinations of operands are supported for VMAD:
            //
            // (U32 * U32) + U32 // intermediate unsigned; final unsigned
            // -(U32 * U32) + S32 // intermediate signed; final signed
            // (U32 * U32) - U32 // intermediate unsigned; final signed
            // (U32 * S32) + S32 // intermediate signed; final signed
            // -(U32 * S32) + S32 // intermediate signed; final signed
            // (U32 * S32) - S32 // intermediate signed; final signed
            // (S32 * U32) + S32 // intermediate signed; final signed
            // -(S32 * U32) + S32 // intermediate signed; final signed
            // (S32 * U32) - S32 // intermediate signed; final signed
            // (S32 * S32) + S32 // intermediate signed; final signed
            // -(S32 * S32) + S32 // intermediate signed; final signed
            // (S32 * S32) - S32 // intermediate signed; final signed

            if (po)
            {
                is_reg(d, dtype).AssertTrue();
                is_alu(a, atype, sel | neg).AssertTrue();
                is_alu(b, btype, sel | neg).AssertTrue();
                is_alu(c, dtype, neg).AssertTrue();
            }
            else
            {
                is_reg(d, dtype).AssertTrue();
                is_alu(a, atype, sel).AssertTrue();
                is_alu(b, btype, sel).AssertTrue();
                is_alu(c, dtype).AssertTrue();
            }
        }
    }
}