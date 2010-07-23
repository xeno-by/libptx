using XenoGears.Assertions;
using Libptx.Edsl.Expressions.Vars;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Vectors
{
    public class v4_s8 : vector
    {
        public v4_s8(reg_s8 x, reg_s8 y, reg_s8 z, reg_s8 w)
        {
            ElementType = s8;
            Elements.Add(x.AssertCast<var>());
            Elements.Add(y.AssertCast<var>());
            Elements.Add(z.AssertCast<var>());
            Elements.Add(w.AssertCast<var>());
        }

        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_u8(v4_s8 v4_s8) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_u8(v4_s8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_s8(v4_s8 v4_s8) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_s8(v4_s8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_b8(v4_s8 v4_s8) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_b8(v4_s8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u8(v4_s8 v4_s8) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u8(v4_s8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s8(v4_s8 v4_s8) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s8(v4_s8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b8(v4_s8 v4_s8) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b8(v4_s8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_u8(v4_s8 v4_s8) { return new Libptx.Edsl.Common.Types.Vector.v4_u8(v4_s8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_s8(v4_s8 v4_s8) { return new Libptx.Edsl.Common.Types.Vector.v4_s8(v4_s8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_b8(v4_s8 v4_s8) { return new Libptx.Edsl.Common.Types.Vector.v4_b8(v4_s8); }
    }
}
