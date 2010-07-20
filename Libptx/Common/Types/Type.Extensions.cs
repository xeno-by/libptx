using System;
using XenoGears.Assertions;

namespace Libptx.Common.Types
{
    public static class TypeExtensions
    {
        public static bool isscalar(this Type type) { return type.kind() != 0; }
        public static bool isint(this Type type) { return (type.kind() & TypeKind.Integer) == TypeKind.Integer; }
        public static bool issigned(this Type type) { return (type.kind() & TypeKind.Signed) == TypeKind.Signed; }
        public static bool isunsigned(this Type type) { return (type.kind() & TypeKind.Unsigned) == TypeKind.Unsigned; }
        public static bool isfloat(this Type type) { return (type.kind() & TypeKind.Float) == TypeKind.Float; }
        public static bool isbit(this Type type) { return (type.kind() & TypeKind.Bit) == TypeKind.Bit; }
        public static bool ispred(this Type type) { return (type.kind() & TypeKind.Pred) == TypeKind.Pred; }
        public static bool istex(this Type type) { return (type.kind() & TypeKind.Tex) == TypeKind.Tex; }
        public static bool issampler(this Type type) { return (type.kind() & TypeKind.Sampler) == TypeKind.Sampler; }
        public static bool issurf(this Type type) { return (type.kind() & TypeKind.Surf) == TypeKind.Surf; }

        [Flags]
        private enum TypeKind
        {
            Integer = 1,
            Signed = 2 | Integer,
            Unsigned = 4 | Integer,
            Float = 8,
            Bit = 16,
            Pred = 32,
            Tex = 64,
            Sampler = 128,
            Surf =256,
        }

        private static TypeKind kind(this Type type)
        {
            if (type == null) return 0;
            if (type.Mod != TypeMod.Scalar) return 0;
            switch (type.Name)
            {
                case TypeName.U8:
                    return TypeKind.Unsigned;
                case TypeName.S8:
                    return TypeKind.Signed;
                case TypeName.U16:
                    return TypeKind.Unsigned;
                case TypeName.S16:
                    return TypeKind.Signed;
                case TypeName.U32:
                    return TypeKind.Unsigned;
                case TypeName.S32:
                    return TypeKind.Signed;
                case TypeName.U64:
                    return TypeKind.Unsigned;
                case TypeName.S64:
                    return TypeKind.Signed;
                case TypeName.F16:
                    return TypeKind.Float;
                case TypeName.F32:
                    return TypeKind.Float;
                case TypeName.F64:
                    return TypeKind.Float;
                case TypeName.B8:
                    return TypeKind.Bit;
                case TypeName.B16:
                    return TypeKind.Bit;
                case TypeName.B32:
                    return TypeKind.Bit;
                case TypeName.B64:
                    return TypeKind.Bit;
                case TypeName.Pred:
                    return TypeKind.Pred;
                case TypeName.Tex:
                    return TypeKind.Tex;
                case TypeName.Sampler:
                    return TypeKind.Sampler;
                case TypeName.Surf:
                    return TypeKind.Surf;
                default:
                    throw AssertionHelper.Fail();
            }
        }

        public static bool is8(this Type type) { return type.bits() == 8; }
        public static bool is16(this Type type) { return type.bits() == 16; }
        public static bool is32(this Type type) { return type.bits() == 32; }
        public static bool is64(this Type type) { return type.bits() == 64; }
        public static int bits(this Type type)
        {
            if (type == null) return 0;
            switch (type.Name)
            {
                case TypeName.U8:
                    return 8;
                case TypeName.S8:
                    return 8;
                case TypeName.U16:
                    return 16;
                case TypeName.S16:
                    return 16;
                case TypeName.U32:
                    return 32;
                case TypeName.S32:
                    return 32;
                case TypeName.U64:
                    return 64;
                case TypeName.S64:
                    return 64;
                case TypeName.F16:
                    return 16;
                case TypeName.F32:
                    return 32;
                case TypeName.F64:
                    return 64;
                case TypeName.B8:
                    return 8;
                case TypeName.B16:
                    return 16;
                case TypeName.B32:
                    return 32;
                case TypeName.B64:
                    return 64;
                case TypeName.Pred:
                    return 1;
                case TypeName.Tex:
                    return 0;
                case TypeName.Sampler:
                    return 0;
                case TypeName.Surf:
                    return 0;
                default:
                    throw AssertionHelper.Fail();
            }
        }

        public static bool isvec(this Type type) { return type.vec() != 0; }
        public static bool isv1(this Type type) { return type.vec() == 1; }
        public static bool isv2(this Type type) { return type.vec() == 2; }
        public static bool isv4(this Type type) { return type.vec() == 4; }
        public static int vec(this Type type)
        {
            if (type == null) return 0;
            if (type.isarr()) return 0;
            if ((type.Mod & TypeMod.V1) == TypeMod.V1) return 1;
            if ((type.Mod & TypeMod.V2) == TypeMod.V2) return 2;
            if ((type.Mod & TypeMod.V4) == TypeMod.V4) return 4;
            return 0;
        }

        public static bool isarr(this Type type) { return type.rank() != 0; }
        public static int rank(this Type type)
        {
            if (type == null) return 0;
            return type.Dims == null ? 0 : type.Dims.Length;
        }
    }
}