using XenoGears.Assertions;
using Libptx.Edsl.Expressions.Vars;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Vectors
{
    public class v4_f16 : vector
    {
        public v4_f16(reg_f16 x, reg_f16 y, reg_f16 z, reg_f16 w)
        {
            ElementType = f16;
            Elements.Add(x.AssertCast<var>());
            Elements.Add(y.AssertCast<var>());
            Elements.Add(z.AssertCast<var>());
            Elements.Add(w.AssertCast<var>());
        }

        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_f16(v4_f16 v4_f16) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_f16(v4_f16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_b16(v4_f16 v4_f16) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_b16(v4_f16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_f16(v4_f16 v4_f16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_f16(v4_f16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b16(v4_f16 v4_f16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b16(v4_f16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b8(v4_f16 v4_f16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b8(v4_f16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_f16(v4_f16 v4_f16) { return new Libptx.Edsl.Common.Types.Vector.v4_f16(v4_f16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_b16(v4_f16 v4_f16) { return new Libptx.Edsl.Common.Types.Vector.v4_b16(v4_f16); }
    }
}
