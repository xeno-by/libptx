using XenoGears.Assertions;
using Libptx.Edsl.Expressions.Slots;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Immediate.Vectors
{
    public class v4_b8 : vector
    {
        public v4_b8(reg_b8 x, reg_b8 y, reg_b8 z, reg_b8 w)
        {
            ElementType = b8;
            Elements.Add(x.AssertCast<var>());
            Elements.Add(y.AssertCast<var>());
            Elements.Add(z.AssertCast<var>());
            Elements.Add(w.AssertCast<var>());
        }

        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_u8(v4_b8 v4_b8) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_u8(v4_b8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_s8(v4_b8 v4_b8) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_s8(v4_b8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_b8(v4_b8 v4_b8) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_b8(v4_b8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u8(v4_b8 v4_b8) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u8(v4_b8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s8(v4_b8 v4_b8) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s8(v4_b8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b8(v4_b8 v4_b8) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b8(v4_b8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_u8(v4_b8 v4_b8) { return new Libptx.Edsl.Common.Types.Vector.v4_u8(v4_b8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_s8(v4_b8 v4_b8) { return new Libptx.Edsl.Common.Types.Vector.v4_s8(v4_b8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_b8(v4_b8 v4_b8) { return new Libptx.Edsl.Common.Types.Vector.v4_b8(v4_b8); }
    }
}
