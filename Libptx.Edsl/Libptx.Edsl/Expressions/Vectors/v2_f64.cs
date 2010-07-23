using XenoGears.Assertions;
using Libptx.Edsl.Expressions.Vars;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Vectors
{
    public class v2_f64 : vector
    {
        public v2_f64(reg_f64 x, reg_f64 y)
        {
            ElementType = f64;
            Elements.Add(x.AssertCast<var>());
            Elements.Add(y.AssertCast<var>());
        }

        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_f64(v2_f64 v2_f64) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_f64(v2_f64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_b64(v2_f64 v2_f64) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_b64(v2_f64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_f64(v2_f64 v2_f64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_f64(v2_f64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_f32(v2_f64 v2_f64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_f32(v2_f64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_f16(v2_f64 v2_f64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_f16(v2_f64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b64(v2_f64 v2_f64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b64(v2_f64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b32(v2_f64 v2_f64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b32(v2_f64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b16(v2_f64 v2_f64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b16(v2_f64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b8(v2_f64 v2_f64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b8(v2_f64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_f64(v2_f64 v2_f64) { return new Libptx.Edsl.Common.Types.Vector.v2_f64(v2_f64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_b64(v2_f64 v2_f64) { return new Libptx.Edsl.Common.Types.Vector.v2_b64(v2_f64); }
    }
}
