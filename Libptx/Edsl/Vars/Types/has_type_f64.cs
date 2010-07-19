using Libptx.Edsl.Types;
using Libptx.Edsl.Types.Relaxed;

namespace Libptx.Edsl.Vars.Types
{
    public class has_type_f64 : has_type<f64>
    {
        public static implicit operator f64(has_type_f64 var_f64) { return new f64{var = var_f64}; }
        public static implicit operator b64(has_type_f64 var_f64) { return new b64{var = var_f64}; }
        public static implicit operator relaxed_f64(has_type_f64 var_f64) { return new relaxed_f64{var = var_f64}; }
        public static implicit operator relaxed_f32(has_type_f64 var_f64) { return new relaxed_f32{var = var_f64}; }
        public static implicit operator relaxed_f16(has_type_f64 var_f64) { return new relaxed_f16{var = var_f64}; }
        public static implicit operator relaxed_b64(has_type_f64 var_f64) { return new relaxed_b64{var = var_f64}; }
        public static implicit operator relaxed_b32(has_type_f64 var_f64) { return new relaxed_b32{var = var_f64}; }
        public static implicit operator relaxed_b16(has_type_f64 var_f64) { return new relaxed_b16{var = var_f64}; }
        public static implicit operator relaxed_b8(has_type_f64 var_f64) { return new relaxed_b8{var = var_f64}; }
    }
}
