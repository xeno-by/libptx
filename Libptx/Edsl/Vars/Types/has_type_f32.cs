using Libptx.Edsl.Types;
using Libptx.Edsl.Types.Relaxed;

namespace Libptx.Edsl.Vars.Types
{
    public class has_type_f32 : has_type<f32>
    {
        public static implicit operator f32(has_type_f32 var_f32) { return new f32{var = var_f32}; }
        public static implicit operator b32(has_type_f32 var_f32) { return new b32{var = var_f32}; }
        public static implicit operator relaxed_f32(has_type_f32 var_f32) { return new relaxed_f32{var = var_f32}; }
        public static implicit operator relaxed_f16(has_type_f32 var_f32) { return new relaxed_f16{var = var_f32}; }
        public static implicit operator relaxed_b32(has_type_f32 var_f32) { return new relaxed_b32{var = var_f32}; }
        public static implicit operator relaxed_b16(has_type_f32 var_f32) { return new relaxed_b16{var = var_f32}; }
        public static implicit operator relaxed_b8(has_type_f32 var_f32) { return new relaxed_b8{var = var_f32}; }
    }
}
