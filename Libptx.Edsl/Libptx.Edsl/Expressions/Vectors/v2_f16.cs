using XenoGears.Assertions;
using Libptx.Edsl.Expressions.Vars;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Vectors
{
    public class v2_f16 : vector
    {
        public v2_f16(reg_f16 x, reg_f16 y)
        {
            ElementType = f16;
            Elements.Add(x.AssertCast<var>());
            Elements.Add(y.AssertCast<var>());
        }

        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_f16(v2_f16 v2_f16) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_f16(v2_f16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_b16(v2_f16 v2_f16) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_b16(v2_f16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_f16(v2_f16 v2_f16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_f16(v2_f16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b16(v2_f16 v2_f16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b16(v2_f16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b8(v2_f16 v2_f16) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b8(v2_f16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_f16(v2_f16 v2_f16) { return new Libptx.Edsl.Common.Types.Vector.v2_f16(v2_f16); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_b16(v2_f16 v2_f16) { return new Libptx.Edsl.Common.Types.Vector.v2_b16(v2_f16); }
    }
}
