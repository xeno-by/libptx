using XenoGears.Assertions;
using Libptx.Edsl.Expressions.Slots;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Immediate.Vectors
{
    public class v2_s32 : vector
    {
        public v2_s32(reg_s32 x, reg_s32 y)
        {
            ElementType = s32;
            Elements.Add(x.AssertCast<var>());
            Elements.Add(y.AssertCast<var>());
        }

        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_u32(v2_s32 v2_s32) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_u32(v2_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_s32(v2_s32 v2_s32) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_s32(v2_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_b32(v2_s32 v2_s32) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_b32(v2_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u32(v2_s32 v2_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u32(v2_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u16(v2_s32 v2_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u16(v2_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u8(v2_s32 v2_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u8(v2_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s32(v2_s32 v2_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s32(v2_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s16(v2_s32 v2_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s16(v2_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s8(v2_s32 v2_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s8(v2_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b32(v2_s32 v2_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b32(v2_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b16(v2_s32 v2_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b16(v2_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b8(v2_s32 v2_s32) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b8(v2_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_u32(v2_s32 v2_s32) { return new Libptx.Edsl.Common.Types.Vector.v2_u32(v2_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_s32(v2_s32 v2_s32) { return new Libptx.Edsl.Common.Types.Vector.v2_s32(v2_s32); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_b32(v2_s32 v2_s32) { return new Libptx.Edsl.Common.Types.Vector.v2_b32(v2_s32); }
    }
}
