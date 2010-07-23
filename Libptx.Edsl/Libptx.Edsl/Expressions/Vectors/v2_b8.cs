using XenoGears.Assertions;
using Libptx.Edsl.Expressions.Vars;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Vectors
{
    public class v2_b8 : vector
    {
        public v2_b8(reg_b8 x, reg_b8 y)
        {
            ElementType = b8;
            Elements.Add(x.AssertCast<var>());
            Elements.Add(y.AssertCast<var>());
        }

        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_u8(v2_b8 v2_b8) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_u8(v2_b8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_s8(v2_b8 v2_b8) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_s8(v2_b8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_b8(v2_b8 v2_b8) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_b8(v2_b8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u8(v2_b8 v2_b8) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u8(v2_b8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s8(v2_b8 v2_b8) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s8(v2_b8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b8(v2_b8 v2_b8) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b8(v2_b8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_u8(v2_b8 v2_b8) { return new Libptx.Edsl.Common.Types.Vector.v2_u8(v2_b8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_s8(v2_b8 v2_b8) { return new Libptx.Edsl.Common.Types.Vector.v2_s8(v2_b8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_b8(v2_b8 v2_b8) { return new Libptx.Edsl.Common.Types.Vector.v2_b8(v2_b8); }
    }
}
