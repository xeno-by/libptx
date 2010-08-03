using XenoGears.Assertions;
using Libptx.Edsl.Expressions.Slots;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Immediate.Vectors
{
    public class v2_f32 : vector
    {
        public v2_f32(reg_f32 x, reg_f32 y)
        {
            ElementType = f32;
            Elements.Add(x.AssertCast<var>());
            Elements.Add(y.AssertCast<var>());
        }

        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_f32(v2_f32 v2_f32) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_f32(v2_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_b32(v2_f32 v2_f32) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_b32(v2_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_f32(v2_f32 v2_f32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_f32(v2_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_f16(v2_f32 v2_f32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_f16(v2_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b32(v2_f32 v2_f32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b32(v2_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b16(v2_f32 v2_f32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b16(v2_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b8(v2_f32 v2_f32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b8(v2_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_f32(v2_f32 v2_f32) { return new Libptx.Edsl.Common.Types.Vector.v2_f32(v2_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_b32(v2_f32 v2_f32) { return new Libptx.Edsl.Common.Types.Vector.v2_b32(v2_f32); }
    }
}
