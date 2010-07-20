using Libptx.Edsl.Types;
using Libptx.Edsl.Types.Relaxed;

namespace Libptx.Edsl.Vars.Types
{
    public class has_type_b16 : has_type<b16>
    {
        public static implicit operator u16(has_type_b16 var_b16) { return new u16{var = var_b16}; }
        public static implicit operator s16(has_type_b16 var_b16) { return new s16{var = var_b16}; }
        public static implicit operator f16(has_type_b16 var_b16) { return new f16{var = var_b16}; }
        public static implicit operator b16(has_type_b16 var_b16) { return new b16{var = var_b16}; }
        public static implicit operator relaxed_u16(has_type_b16 var_b16) { return new relaxed_u16{var = var_b16}; }
        public static implicit operator relaxed_u8(has_type_b16 var_b16) { return new relaxed_u8{var = var_b16}; }
        public static implicit operator relaxed_s16(has_type_b16 var_b16) { return new relaxed_s16{var = var_b16}; }
        public static implicit operator relaxed_s8(has_type_b16 var_b16) { return new relaxed_s8{var = var_b16}; }
        public static implicit operator relaxed_f16(has_type_b16 var_b16) { return new relaxed_f16{var = var_b16}; }
        public static implicit operator relaxed_b16(has_type_b16 var_b16) { return new relaxed_b16{var = var_b16}; }
        public static implicit operator relaxed_b8(has_type_b16 var_b16) { return new relaxed_b8{var = var_b16}; }
    }
}
