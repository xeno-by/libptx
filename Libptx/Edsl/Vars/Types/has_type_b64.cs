using Libptx.Edsl.Types;
using Libptx.Edsl.Types.Relaxed;

namespace Libptx.Edsl.Vars.Types
{
    public class has_type_b64 : has_type<b64>
    {
        public static implicit operator u64(has_type_b64 var_b64) { return new u64{var = var_b64}; }
        public static implicit operator s64(has_type_b64 var_b64) { return new s64{var = var_b64}; }
        public static implicit operator f64(has_type_b64 var_b64) { return new f64{var = var_b64}; }
        public static implicit operator b64(has_type_b64 var_b64) { return new b64{var = var_b64}; }
        public static implicit operator relaxed_u64(has_type_b64 var_b64) { return new relaxed_u64{var = var_b64}; }
        public static implicit operator relaxed_u32(has_type_b64 var_b64) { return new relaxed_u32{var = var_b64}; }
        public static implicit operator relaxed_u16(has_type_b64 var_b64) { return new relaxed_u16{var = var_b64}; }
        public static implicit operator relaxed_u8(has_type_b64 var_b64) { return new relaxed_u8{var = var_b64}; }
        public static implicit operator relaxed_s64(has_type_b64 var_b64) { return new relaxed_s64{var = var_b64}; }
        public static implicit operator relaxed_s32(has_type_b64 var_b64) { return new relaxed_s32{var = var_b64}; }
        public static implicit operator relaxed_s16(has_type_b64 var_b64) { return new relaxed_s16{var = var_b64}; }
        public static implicit operator relaxed_s8(has_type_b64 var_b64) { return new relaxed_s8{var = var_b64}; }
        public static implicit operator relaxed_f64(has_type_b64 var_b64) { return new relaxed_f64{var = var_b64}; }
        public static implicit operator relaxed_f32(has_type_b64 var_b64) { return new relaxed_f32{var = var_b64}; }
        public static implicit operator relaxed_f16(has_type_b64 var_b64) { return new relaxed_f16{var = var_b64}; }
        public static implicit operator relaxed_b64(has_type_b64 var_b64) { return new relaxed_b64{var = var_b64}; }
        public static implicit operator relaxed_b32(has_type_b64 var_b64) { return new relaxed_b32{var = var_b64}; }
        public static implicit operator relaxed_b16(has_type_b64 var_b64) { return new relaxed_b16{var = var_b64}; }
        public static implicit operator relaxed_b8(has_type_b64 var_b64) { return new relaxed_b8{var = var_b64}; }
    }
}
