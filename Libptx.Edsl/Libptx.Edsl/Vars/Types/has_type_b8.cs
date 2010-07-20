using Libptx.Edsl.Types;
using Libptx.Edsl.Types.Relaxed;

namespace Libptx.Edsl.Vars.Types
{
    public class has_type_b8 : has_type<b8>
    {
        public static implicit operator u8(has_type_b8 var_b8) { return new u8{var = var_b8}; }
        public static implicit operator s8(has_type_b8 var_b8) { return new s8{var = var_b8}; }
        public static implicit operator b8(has_type_b8 var_b8) { return new b8{var = var_b8}; }
        public static implicit operator relaxed_u8(has_type_b8 var_b8) { return new relaxed_u8{var = var_b8}; }
        public static implicit operator relaxed_s8(has_type_b8 var_b8) { return new relaxed_s8{var = var_b8}; }
        public static implicit operator relaxed_b8(has_type_b8 var_b8) { return new relaxed_b8{var = var_b8}; }
    }
}
