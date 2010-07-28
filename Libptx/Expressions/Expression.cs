using System.Diagnostics;
using Libptx.Common;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Expressions.Addresses;
using Libptx.Expressions.Slots;
using XenoGears.Assertions;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions
{
    public interface Expression : Validatable, Renderable
    {
        Type Type { get; }
    }

    [DebuggerNonUserCode]
    public static class ExpressionExtensions
    {
        public static Modded mod(this Expression expr, Mod mod) { return new Modded{Expr = expr, Mod = mod}; }
        public static Mod exact(this Mod mod) { return mod | (Mod)65536; }
        public static bool has_mod(this Expression expr, Mod mod)
        {
            var is_exact = ((int)mod & 65536) == 65536;
            mod = (Mod)((int)mod & ~65536);

            var modded = expr as Modded;
            if (modded != null)
            {
                if (is_exact) return modded.Mod == mod;
                else return mod == 0 || (modded.Mod & ~mod) == 0;
            }
            else
            {
                if (is_exact) return mod == 0;
                else return true;
            }
        }

        public static bool is_scalar(this Expression expr) { return expr == null ? false : expr.Type.is_scalar(); }
        public static bool is_int(this Expression expr) { return expr == null ? false : expr.Type.is_int(); }
        public static bool is_signed(this Expression expr) { return expr == null ? false : expr.Type.is_signed(); }
        public static bool is_unsigned(this Expression expr) { return expr == null ? false : expr.Type.is_unsigned(); }
        public static bool is_float(this Expression expr) { return expr == null ? false : expr.Type.is_float(); }
        public static bool is_bit(this Expression expr) { return expr == null ? false : expr.Type.is_bit(); }

        public static bool is_pred(this Expression expr) { return expr == null ? false : expr.Type.is_pred(); }
        public static bool is_ptr(this Expression expr) { return expr == null ? false : expr.Type.is_ptr(); }
        public static bool is_ptr(this Expression expr, space space)
        {
            if (!expr.is_ptr()) return false;
            if (space == 0) return true;

            var e_var = expr as Var;
            if (e_var != null) return (e_var.Space & ~space) == 0;

            var e_addr = expr as Address;
            if (e_addr != null)
            {
                var ok = true;

                var a_base = e_addr.Base as Var;
                if (a_base != null) ok &= ((a_base.Space & ~space) == 0);

                var o_base = e_addr.Offset.Base as Var;
                if (o_base != null) ok &= ((o_base.Space & ~space) == 0);

                return ok;
            }

            throw AssertionHelper.Fail();
        }
        public static bool is_bmk(this Expression expr) { return expr == null ? false : expr.Type.is_bmk(); }

        public static bool is_opaque(this Expression expr) { return expr == null ? false : expr.Type.is_opaque(); }
        public static bool is_texref(this Expression expr) { return expr == null ? false : expr.Type.is_texref(); }
        public static bool is_samplerref(this Expression expr) { return expr == null ? false : expr.Type.is_samplerref(); }
        public static bool is_surfref(this Expression expr) { return expr == null ? false : expr.Type.is_surfref(); }

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