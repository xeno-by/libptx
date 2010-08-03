using XenoGears.Assertions;
using Libptx.Edsl.Expressions.Slots;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Immediate.Vectors
{
    public class v1_u8 : vector
    {
        public v1_u8(reg_u8 x)
        {
            ElementType = u8;
            Elements.Add(x.AssertCast<var>());
        }

        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v1_u8(v1_u8 v1_u8) { return new Libptx.Edsl.Common.Types.Vector.reg_v1_u8(v1_u8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v1_s8(v1_u8 v1_u8) { return new Libptx.Edsl.Common.Types.Vector.reg_v1_s8(v1_u8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v1_b8(v1_u8 v1_u8) { return new Libptx.Edsl.Common.Types.Vector.reg_v1_b8(v1_u8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v1_u8(v1_u8 v1_u8) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v1_u8(v1_u8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v1_s8(v1_u8 v1_u8) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v1_s8(v1_u8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v1_b8(v1_u8 v1_u8) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v1_b8(v1_u8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v1_u8(v1_u8 v1_u8) { return new Libptx.Edsl.Common.Types.Vector.v1_u8(v1_u8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v1_s8(v1_u8 v1_u8) { return new Libptx.Edsl.Common.Types.Vector.v1_s8(v1_u8); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v1_b8(v1_u8 v1_u8) { return new Libptx.Edsl.Common.Types.Vector.v1_b8(v1_u8); }
    }
}
