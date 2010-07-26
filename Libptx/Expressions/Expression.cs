using Libptx.Common.Types;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions
{
    public interface Expression
    {
        Type Type { get; }
    }

    public static class ExpressionExtensions
    {
        public static Modded mod(this Expression expr, Mod mod)
        {
            return new Modded{Expr = expr, Mod = mod};
        }

        public static bool is_opaque(this Expression expr) { return expr == null ? false : expr.Type.is_opaque(); }
        public static bool is_pred(this Expression expr) { return expr == null ? false : expr.Type.is_pred(); }
        public static bool is_texref(this Expression expr) { return expr == null ? false : expr.Type.is_texref(); }
        public static bool is_samplerref(this Expression expr) { return expr == null ? false : expr.Type.is_samplerref(); }
        public static bool is_surfref(this Expression expr) { return expr == null ? false : expr.Type.is_surfref(); }

        public static bool is_scalar(this Expression expr) { return expr == null ? false : expr.Type.is_scalar(); }
        public static bool is_int(this Expression expr) { return expr == null ? false : expr.Type.is_int(); }
        public static bool is_signed(this Expression expr) { return expr == null ? false : expr.Type.is_signed(); }
        public static bool is_unsigned(this Expression expr) { return expr == null ? false : expr.Type.is_unsigned(); }
        public static bool is_float(this Expression expr) { return expr == null ? false : expr.Type.is_float(); }
        public static bool is_bit(this Expression expr) { return expr == null ? false : expr.Type.is_bit(); }

        public static bool is_vec(this Expression expr) { return expr == null ? false : expr.Type.is_vec(); }
        public static int vec_rank(this Expression expr) { return expr == null ? 0 : expr.Type.vec_rank(); }
        public static Type vec_el(this Expression expr) { return expr == null ? null : expr.Type.vec_el(); }
        public static bool is_v1(this Expression expr) { return expr == null ? false : expr.Type.is_v1(); }
        public static bool is_v2(this Expression expr) { return expr == null ? false : expr.Type.is_v2(); }
        public static bool is_v4(this Expression expr) { return expr == null ? false : expr.Type.is_v4(); }

        public static bool is_arr(this Expression expr) { return expr == null ? false : expr.Type.is_arr(); }
        public static int arr_rank(this Expression expr) { return expr == null ? 0 : expr.Type.arr_rank(); }
        public static Type arr_el(this Expression expr) { return expr == null ? null : expr.Type.arr_el(); }

        public static bool is8(this Expression expr) { return expr == null ? false : expr.Type.is8(); }
        public static bool is16(this Expression expr) { return expr == null ? false : expr.Type.is16(); }
        public static bool is32(this Expression expr) { return expr == null ? false : expr.Type.is32(); }
        public static bool is64(this Expression expr) { return expr == null ? false : expr.Type.is64(); }
        public static int bits(this Expression expr) { return expr == null ? 0 : expr.Type.bits(); }
    }
}