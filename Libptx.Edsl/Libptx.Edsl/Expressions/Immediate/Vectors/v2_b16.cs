using XenoGears.Assertions;
using Libptx.Edsl.Expressions.Slots;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Immediate.Vectors
{
    public class v2_b16 : vector
    {
        public v2_b16(reg_b16 x, reg_b16 y)
        {
            ElementType = b16;
            Elements.Add(x.AssertCast<var>());
            Elements.Add(y.AssertCast<var>());
        }

        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_u16(v2_b16 v2_b16) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_u16(v2_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_s16(v2_b16 v2_b16) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_s16(v2_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_f16(v2_b16 v2_b16) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_f16(v2_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_b16(v2_b16 v2_b16) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_b16(v2_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u16(v2_b16 v2_b16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u16(v2_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u8(v2_b16 v2_b16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u8(v2_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s16(v2_b16 v2_b16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s16(v2_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s8(v2_b16 v2_b16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s8(v2_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_f16(v2_b16 v2_b16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_f16(v2_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b16(v2_b16 v2_b16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b16(v2_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b8(v2_b16 v2_b16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b8(v2_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_u16(v2_b16 v2_b16) { return new Libptx.Edsl.Common.Types.Vector.v2_u16(v2_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_s16(v2_b16 v2_b16) { return new Libptx.Edsl.Common.Types.Vector.v2_s16(v2_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_f16(v2_b16 v2_b16) { return new Libptx.Edsl.Common.Types.Vector.v2_f16(v2_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_b16(v2_b16 v2_b16) { return new Libptx.Edsl.Common.Types.Vector.v2_b16(v2_b16); }
    }
}
