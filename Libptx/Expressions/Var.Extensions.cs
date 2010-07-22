using Libptx.Common.Types;

namespace Libptx.Expressions
{
    public static class VarExtensions
    {
        public static bool isscalar(this Var var) { return var == null ? false : var.Type.isscalar(); }
        public static bool isint(this Var var) { return var == null ? false : var.Type.isint(); }
        public static bool issigned(this Var var) { return var == null ? false : var.Type.issigned(); }
        public static bool isunsigned(this Var var) { return var == null ? false : var.Type.isunsigned(); }
        public static bool isfloat(this Var var) { return var == null ? false : var.Type.isfloat(); }
        public static bool isbit(this Var var) { return var == null ? false : var.Type.isbit(); }
        public static bool ispred(this Var var) { return var == null ? false : var.Type.ispred(); }
        public static bool istex(this Var var) { return var == null ? false : var.Type.istex(); }
        public static bool issampler(this Var var) { return var == null ? false : var.Type.issampler(); }
        public static bool issurf(this Var var) { return var == null ? false : var.Type.issurf(); }

        public static bool is8(this Var var) { return var == null ? false : var.Type.is8(); }
        public static bool is16(this Var var) { return var == null ? false : var.Type.is16(); }
        public static bool is32(this Var var) { return var == null ? false : var.Type.is32(); }
        public static bool is64(this Var var) { return var == null ? false : var.Type.is64(); }
        public static int bits(this Var var) { return var == null ? 0 : var.Type.bits(); }

        public static bool isvec(this Var var) { return var == null ? false : var.Type.isvec(); }
        public static bool isv1(this Var var) { return var == null ? false : var.Type.isv1(); }
        public static bool isv2(this Var var) { return var == null ? false : var.Type.isv2(); }
        public static bool isv4(this Var var) { return var == null ? false : var.Type.isv4(); }
        public static int vec(this Var var) { return var == null ? 0 : var.Type.vec(); }

        public static bool isarr(this Var var) { return var == null ? false : var.Type.isarr(); }
        public static int rank(this Var var) { return var == null ? 0 : var.Type.rank(); }
    }
}