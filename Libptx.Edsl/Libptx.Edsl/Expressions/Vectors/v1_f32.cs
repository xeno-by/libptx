using XenoGears.Assertions;
using Libptx.Edsl.Expressions.Vars;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Vectors
{
    public class v1_f32 : vector
    {
        public v1_f32(reg_f32 x)
        {
            ElementType = f32;
            Elements.Add(x.AssertCast<var>());
        }

        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v1_f32(v1_f32 v1_f32) { return new Libptx.Edsl.Common.Types.Vector.reg_v1_f32(v1_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v1_b32(v1_f32 v1_f32) { return new Libptx.Edsl.Common.Types.Vector.reg_v1_b32(v1_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v1_f32(v1_f32 v1_f32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v1_f32(v1_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v1_f16(v1_f32 v1_f32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v1_f16(v1_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v1_b32(v1_f32 v1_f32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v1_b32(v1_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v1_b16(v1_f32 v1_f32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v1_b16(v1_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v1_b8(v1_f32 v1_f32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v1_b8(v1_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v1_f32(v1_f32 v1_f32) { return new Libptx.Edsl.Common.Types.Vector.v1_f32(v1_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v1_b32(v1_f32 v1_f32) { return new Libptx.Edsl.Common.Types.Vector.v1_b32(v1_f32); }
    }
}
