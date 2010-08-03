using XenoGears.Assertions;
using Libptx.Edsl.Expressions.Slots;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Immediate.Vectors
{
    public class v4_u32 : vector
    {
        public v4_u32(reg_u32 x, reg_u32 y, reg_u32 z, reg_u32 w)
        {
            ElementType = u32;
            Elements.Add(x.AssertCast<var>());
            Elements.Add(y.AssertCast<var>());
            Elements.Add(z.AssertCast<var>());
            Elements.Add(w.AssertCast<var>());
        }

        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_u32(v4_u32 v4_u32) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_u32(v4_u32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_s32(v4_u32 v4_u32) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_s32(v4_u32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v4_b32(v4_u32 v4_u32) { return new Libptx.Edsl.Common.Types.Vector.reg_v4_b32(v4_u32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u32(v4_u32 v4_u32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u32(v4_u32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u16(v4_u32 v4_u32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u16(v4_u32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u8(v4_u32 v4_u32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_u8(v4_u32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s32(v4_u32 v4_u32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s32(v4_u32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s16(v4_u32 v4_u32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s16(v4_u32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s8(v4_u32 v4_u32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_s8(v4_u32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b32(v4_u32 v4_u32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b32(v4_u32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b16(v4_u32 v4_u32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b16(v4_u32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b8(v4_u32 v4_u32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v4_b8(v4_u32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_u32(v4_u32 v4_u32) { return new Libptx.Edsl.Common.Types.Vector.v4_u32(v4_u32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_s32(v4_u32 v4_u32) { return new Libptx.Edsl.Common.Types.Vector.v4_s32(v4_u32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v4_b32(v4_u32 v4_u32) { return new Libptx.Edsl.Common.Types.Vector.v4_b32(v4_u32); }
    }
}
