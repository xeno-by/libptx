using Libptx.Common.Types;

namespace Libptx.Expressions
{
    public static class VarExtensions
    {
        public static bool is_opaque(this Var var) { return var == null ? false : var.Type.is_opaque(); }
        public static bool is_pred(this Var var) { return var == null ? false : var.Type.is_pred(); }
        public static bool is_texref(this Var var) { return var == null ? false : var.Type.is_texref(); }
        public static bool is_samplerref(this Var var) { return var == null ? false : var.Type.is_samplerref(); }
        public static bool is_surfref(this Var var) { return var == null ? false : var.Type.is_surfref(); }

        public static bool is_scalar(this Var var) { return var == null ? false : var.Type.is_scalar(); }
        public static bool is_int(this Var var) { return var == null ? false : var.Type.is_int(); }
        public static bool is_signed(this Var var) { return var == null ? false : var.Type.is_signed(); }
        public static bool is_unsigned(this Var var) { return var == null ? false : var.Type.is_unsigned(); }
        public static bool is_float(this Var var) { return var == null ? false : var.Type.is_float(); }
        public static bool is_bit(this Var var) { return var == null ? false : var.Type.is_bit(); }

        public static bool is_vec(this Var var) { return var == null ? false : var.Type.is_vec(); }
        public static int vec_rank(this Var var) { return var == null ? 0 : var.Type.vec_rank(); }
        public static Type vec_el(this Var var) { return var == null ? null : var.Type.vec_el(); }
        public static bool is_v1(this Var var) { return var == null ? false : var.Type.is_v1(); }
        public static bool is_v2(this Var var) { return var == null ? false : var.Type.is_v2(); }
        public static bool is_v4(this Var var) { return var == null ? false : var.Type.is_v4(); }

        public static bool is_arr(this Var var) { return var == null ? false : var.Type.is_arr(); }
        public static int arr_rank(this Var var) { return var == null ? 0 : var.Type.arr_rank(); }
        public static Type arr_el(this Var var) { return var == null ? null : var.Type.arr_el(); }

        public static bool is8(this Var var) { return var == null ? false : var.Type.is8(); }
        public static bool is16(this Var var) { return var == null ? false : var.Type.is16(); }
        public static bool is32(this Var var) { return var == null ? false : var.Type.is32(); }
        public static bool is64(this Var var) { return var == null ? false : var.Type.is64(); }
        public static int bits(this Var var) { return var == null ? 0 : var.Type.bits(); }
    }
}