using XenoGears.Assertions;
using Libptx.Edsl.Expressions.Vars;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Vectors
{
    public class v4_b16 : vector
    {
        public v4_b16(reg_b16 x, reg_b16 y, reg_b16 z, reg_b16 w)
        {
            Elements.Add(x.AssertCast<var>());
            Elements.Add(y.AssertCast<var>());
            Elements.Add(z.AssertCast<var>());
            Elements.Add(w.AssertCast<var>());
        }

        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_u16(v4_b16 v4_b16) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_u16(v4_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_s16(v4_b16 v4_b16) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_s16(v4_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_f16(v4_b16 v4_b16) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_f16(v4_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_b16(v4_b16 v4_b16) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_b16(v4_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u16(v4_b16 v4_b16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u16(v4_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u8(v4_b16 v4_b16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u8(v4_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s16(v4_b16 v4_b16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s16(v4_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s8(v4_b16 v4_b16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s8(v4_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_f16(v4_b16 v4_b16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_f16(v4_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b16(v4_b16 v4_b16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b16(v4_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b8(v4_b16 v4_b16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b8(v4_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_u16(v4_b16 v4_b16) { return new Libptx.Edsl.Common.Types.Vector.v4_u16(v4_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_s16(v4_b16 v4_b16) { return new Libptx.Edsl.Common.Types.Vector.v4_s16(v4_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_f16(v4_b16 v4_b16) { return new Libptx.Edsl.Common.Types.Vector.v4_f16(v4_b16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_b16(v4_b16 v4_b16) { return new Libptx.Edsl.Common.Types.Vector.v4_b16(v4_b16); }
    }
}
