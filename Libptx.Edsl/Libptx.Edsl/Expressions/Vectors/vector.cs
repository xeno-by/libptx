using Libptx.Expressions;
using Libptx.Edsl.Common.Types.Scalar;

namespace Libptx.Edsl.Expressions.Vectors
{
    public class vector : Vector
    {
        public static v1_u8 v1(reg_u8 x) { return new v1_u8(x); }
        public static v1_s8 v1(reg_s8 x) { return new v1_s8(x); }
        public static v1_u16 v1(reg_u16 x) { return new v1_u16(x); }
        public static v1_s16 v1(reg_s16 x) { return new v1_s16(x); }
        public static v1_u32 v1(reg_u32 x) { return new v1_u32(x); }
        public static v1_s32 v1(reg_s32 x) { return new v1_s32(x); }
        public static v1_u64 v1(reg_u64 x) { return new v1_u64(x); }
        public static v1_s64 v1(reg_s64 x) { return new v1_s64(x); }
        public static v1_f16 v1(reg_f16 x) { return new v1_f16(x); }
        public static v1_f32 v1(reg_f32 x) { return new v1_f32(x); }
        public static v1_f64 v1(reg_f64 x) { return new v1_f64(x); }
        public static v1_b8 v1(reg_b8 x) { return new v1_b8(x); }
        public static v1_b16 v1(reg_b16 x) { return new v1_b16(x); }
        public static v1_b32 v1(reg_b32 x) { return new v1_b32(x); }
        public static v1_b64 v1(reg_b64 x) { return new v1_b64(x); }

        public static v2_u8 v2(reg_u8 x, reg_u8 y) { return new v2_u8(x, y); }
        public static v2_s8 v2(reg_s8 x, reg_s8 y) { return new v2_s8(x, y); }
        public static v2_u16 v2(reg_u16 x, reg_u16 y) { return new v2_u16(x, y); }
        public static v2_s16 v2(reg_s16 x, reg_s16 y) { return new v2_s16(x, y); }
        public static v2_u32 v2(reg_u32 x, reg_u32 y) { return new v2_u32(x, y); }
        public static v2_s32 v2(reg_s32 x, reg_s32 y) { return new v2_s32(x, y); }
        public static v2_u64 v2(reg_u64 x, reg_u64 y) { return new v2_u64(x, y); }
        public static v2_s64 v2(reg_s64 x, reg_s64 y) { return new v2_s64(x, y); }
        public static v2_f16 v2(reg_f16 x, reg_f16 y) { return new v2_f16(x, y); }
        public static v2_f32 v2(reg_f32 x, reg_f32 y) { return new v2_f32(x, y); }
        public static v2_f64 v2(reg_f64 x, reg_f64 y) { return new v2_f64(x, y); }
        public static v2_b8 v2(reg_b8 x, reg_b8 y) { return new v2_b8(x, y); }
        public static v2_b16 v2(reg_b16 x, reg_b16 y) { return new v2_b16(x, y); }
        public static v2_b32 v2(reg_b32 x, reg_b32 y) { return new v2_b32(x, y); }
        public static v2_b64 v2(reg_b64 x, reg_b64 y) { return new v2_b64(x, y); }

        public static v4_u8 v4(reg_u8 x, reg_u8 y, reg_u8 z, reg_u8 w) { return new v4_u8(x, y, z, w); }
        public static v4_s8 v4(reg_s8 x, reg_s8 y, reg_s8 z, reg_s8 w) { return new v4_s8(x, y, z, w); }
        public static v4_u16 v4(reg_u16 x, reg_u16 y, reg_u16 z, reg_u16 w) { return new v4_u16(x, y, z, w); }
        public static v4_s16 v4(reg_s16 x, reg_s16 y, reg_s16 z, reg_s16 w) { return new v4_s16(x, y, z, w); }
        public static v4_u32 v4(reg_u32 x, reg_u32 y, reg_u32 z, reg_u32 w) { return new v4_u32(x, y, z, w); }
        public static v4_s32 v4(reg_s32 x, reg_s32 y, reg_s32 z, reg_s32 w) { return new v4_s32(x, y, z, w); }
        public static v4_f16 v4(reg_f16 x, reg_f16 y, reg_f16 z, reg_f16 w) { return new v4_f16(x, y, z, w); }
        public static v4_f32 v4(reg_f32 x, reg_f32 y, reg_f32 z, reg_f32 w) { return new v4_f32(x, y, z, w); }
        public static v4_b8 v4(reg_b8 x, reg_b8 y, reg_b8 z, reg_b8 w) { return new v4_b8(x, y, z, w); }
        public static v4_b16 v4(reg_b16 x, reg_b16 y, reg_b16 z, reg_b16 w) { return new v4_b16(x, y, z, w); }
        public static v4_b32 v4(reg_b32 x, reg_b32 y, reg_b32 z, reg_b32 w) { return new v4_b32(x, y, z, w); }
    }
}
