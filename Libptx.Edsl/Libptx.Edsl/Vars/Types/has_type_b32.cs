using Libptx.Edsl.Types;
using Libptx.Edsl.Types.Relaxed;

namespace Libptx.Edsl.Vars.Types
{
    public class has_type_b32 : has_type<b32>
    {
        public static implicit operator u32(has_type_b32 var_b32) { return new u32{var = var_b32}; }
        public static implicit operator s32(has_type_b32 var_b32) { return new s32{var = var_b32}; }
        public static implicit operator f32(has_type_b32 var_b32) { return new f32{var = var_b32}; }
        public static implicit operator b32(has_type_b32 var_b32) { return new b32{var = var_b32}; }
        public static implicit operator relaxed_u32(has_type_b32 var_b32) { return new relaxed_u32{var = var_b32}; }
        public static implicit operator relaxed_u16(has_type_b32 var_b32) { return new relaxed_u16{var = var_b32}; }
        public static implicit operator relaxed_u8(has_type_b32 var_b32) { return new relaxed_u8{var = var_b32}; }
        public static implicit operator relaxed_s32(has_type_b32 var_b32) { return new relaxed_s32{var = var_b32}; }
        public static implicit operator relaxed_s16(has_type_b32 var_b32) { return new relaxed_s16{var = var_b32}; }
        public static implicit operator relaxed_s8(has_type_b32 var_b32) { return new relaxed_s8{var = var_b32}; }
        public static implicit operator relaxed_f32(has_type_b32 var_b32) { return new relaxed_f32{var = var_b32}; }
        public static implicit operator relaxed_f16(has_type_b32 var_b32) { return new relaxed_f16{var = var_b32}; }
        public static implicit operator relaxed_b32(has_type_b32 var_b32) { return new relaxed_b32{var = var_b32}; }
        public static implicit operator relaxed_b16(has_type_b32 var_b32) { return new relaxed_b16{var = var_b32}; }
        public static implicit operator relaxed_b8(has_type_b32 var_b32) { return new relaxed_b8{var = var_b32}; }
    }
}
