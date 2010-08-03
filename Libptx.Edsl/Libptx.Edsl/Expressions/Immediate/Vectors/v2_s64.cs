using XenoGears.Assertions;
using Libptx.Edsl.Expressions.Slots;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Immediate.Vectors
{
    public class v2_s64 : vector
    {
        public v2_s64(reg_s64 x, reg_s64 y)
        {
            ElementType = s64;
            Elements.Add(x.AssertCast<var>());
            Elements.Add(y.AssertCast<var>());
        }

        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_u64(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_u64(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_s64(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_s64(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.reg_v2_b64(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.reg_v2_b64(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u64(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u64(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u32(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u32(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u16(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u16(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u8(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_u8(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s64(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s64(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s32(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s32(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s16(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s16(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s8(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_s8(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b64(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b64(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b32(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b32(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b16(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b16(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b8(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.relaxed_reg_v2_b8(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_u64(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.v2_u64(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_s64(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.v2_s64(v2_s64); }
        public static implicit operator Libptx.Edsl.Common.Types.Vector.v2_b64(v2_s64 v2_s64) { return new Libptx.Edsl.Common.Types.Vector.v2_b64(v2_s64); }
    }
}
