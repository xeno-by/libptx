using Libptx.Edsl.Types;
using Libptx.Edsl.Types.Relaxed;

namespace Libptx.Edsl.Vars.Types
{
    public class has_type_f16 : has_type<f16>
    {
        public static implicit operator f16(has_type_f16 var_f16) { return new f16{var = var_f16}; }
        public static implicit operator b16(has_type_f16 var_f16) { return new b16{var = var_f16}; }
        public static implicit operator relaxed_f16(has_type_f16 var_f16) { return new relaxed_f16{var = var_f16}; }
        public static implicit operator relaxed_b16(has_type_f16 var_f16) { return new relaxed_b16{var = var_f16}; }
        public static implicit operator relaxed_b8(has_type_f16 var_f16) { return new relaxed_b8{var = var_f16}; }
    }
}
