using XenoGears.Assertions;
using Libptx.Edsl.Expressions.Vars;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Vectors
{
    public class v4_s32 : vector
    {
        public v4_s32(reg_s32 x, reg_s32 y, reg_s32 z, reg_s32 w)
        {
            ElementType = s32;
            Elements.Add(x.AssertCast<var>());
            Elements.Add(y.AssertCast<var>());
            Elements.Add(z.AssertCast<var>());
            Elements.Add(w.AssertCast<var>());
        }

        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_u32(v4_s32 v4_s32) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_u32(v4_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_s32(v4_s32 v4_s32) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_s32(v4_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_b32(v4_s32 v4_s32) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_b32(v4_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u32(v4_s32 v4_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u32(v4_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u16(v4_s32 v4_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u16(v4_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u8(v4_s32 v4_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u8(v4_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s32(v4_s32 v4_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s32(v4_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s16(v4_s32 v4_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s16(v4_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s8(v4_s32 v4_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s8(v4_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b32(v4_s32 v4_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b32(v4_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b16(v4_s32 v4_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b16(v4_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b8(v4_s32 v4_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b8(v4_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_u32(v4_s32 v4_s32) { return new Libptx.Edsl.Common.Types.Vector.v4_u32(v4_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_s32(v4_s32 v4_s32) { return new Libptx.Edsl.Common.Types.Vector.v4_s32(v4_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_b32(v4_s32 v4_s32) { return new Libptx.Edsl.Common.Types.Vector.v4_b32(v4_s32); }
    }
}
