using XenoGears.Assertions;
using Libptx.Edsl.Expressions.Vars;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Vectors
{
    public class v4_f32 : vector
    {
        public v4_f32(reg_f32 x, reg_f32 y, reg_f32 z, reg_f32 w)
        {
            Elements.Add(x.AssertCast<var>());
            Elements.Add(y.AssertCast<var>());
            Elements.Add(z.AssertCast<var>());
            Elements.Add(w.AssertCast<var>());
        }

        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_f32(v4_f32 v4_f32) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_f32(v4_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_b32(v4_f32 v4_f32) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_b32(v4_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_f32(v4_f32 v4_f32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_f32(v4_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_f16(v4_f32 v4_f32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_f16(v4_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b32(v4_f32 v4_f32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b32(v4_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b16(v4_f32 v4_f32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b16(v4_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b8(v4_f32 v4_f32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b8(v4_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_f32(v4_f32 v4_f32) { return new Libptx.Edsl.Common.Types.Vector.v4_f32(v4_f32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_b32(v4_f32 v4_f32) { return new Libptx.Edsl.Common.Types.Vector.v4_b32(v4_f32); }
    }
}
