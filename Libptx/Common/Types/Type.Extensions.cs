using System;
using XenoGears.Assertions;
using XenoGears.Functional;
using System.Linq;

namespace Libptx.Common.Types
{
    public static class TypeExtensions
    {
        [Flags]
        private enum TypeSpec
        {
            Integer = 2,
            Signed = 4 | Integer,
            Unsigned = 8 | Integer,
            Float = 16,
            Bit = 32,
            Opaque = 64,
            Pred = 128 | Opaque,
            Texref = 256 | Opaque,
            Samplerref = 512 | Opaque,
            Surfref = 1024 | Opaque,
            V1 = 2048,
            V2 = 4096,
            V4 = 8192,
            A1 = 16384,
        }

        private static TypeSpec spec(this Type type)
        {
            Func<TypeName, TypeSpec> spec_from_name = name =>
            {
                switch (name)
                {
                    case TypeName.U8:
                        return TypeSpec.Unsigned;
                    case TypeName.S8:
                        return TypeSpec.Signed;
                    case TypeName.U16:
                        return TypeSpec.Unsigned;
                    case TypeName.S16:
                        return TypeSpec.Signed;
                    case TypeName.U32:
                        return TypeSpec.Unsigned;
                    case TypeName.S32:
                        return TypeSpec.Signed;
                    case TypeName.U64:
                        return TypeSpec.Unsigned;
                    case TypeName.S64:
                        return TypeSpec.Signed;
                    case TypeName.F16:
                        return TypeSpec.Float;
                    case TypeName.F32:
                        return TypeSpec.Float;
                    case TypeName.F64:
                        return TypeSpec.Float;
                    case TypeName.B8:
                        return TypeSpec.Bit;
                    case TypeName.B16:
                        return TypeSpec.Bit;
                    case TypeName.B32:
                        return TypeSpec.Bit;
                    case TypeName.B64:
                        return TypeSpec.Bit;
                    case TypeName.Pred:
                        return TypeSpec.Pred;
                    case TypeName.Texref:
                        return TypeSpec.Texref;
                    case TypeName.Samplerref:
                        return TypeSpec.Samplerref;
                    case TypeName.Surfref:
                        return TypeSpec.Surfref;
                    default:
                        throw AssertionHelper.Fail();
                }
            };

            if (type == null) return 0;
            var spec = spec_from_name(type.Name);
            if ((type.Mod & TypeMod.V1) == TypeMod.V1) spec |= TypeSpec.V1;
            if ((type.Mod & TypeMod.V2) == TypeMod.V2) spec |= TypeSpec.V2;
            if ((type.Mod & TypeMod.V4) == TypeMod.V4) spec |= TypeSpec.V4;
            if ((type.Dims ?? Seq.Empty<int>()).Count() == 1) spec |= TypeSpec.A1;
            return spec;
        }

        public static bool is_opaque(this Type type) { return (type.spec() & TypeSpec.Opaque) == TypeSpec.Opaque; }
        public static bool is_pred(this Type type) { return (type.spec() & TypeSpec.Pred) == TypeSpec.Pred; }
        public static bool is_texref(this Type type) { return (type.spec() & TypeSpec.Texref) == TypeSpec.Texref; }
        public static bool is_samplerref(this Type type) { return (type.spec() & TypeSpec.Samplerref) == TypeSpec.Samplerref; }
        public static bool is_surfref(this Type type) { return (type.spec() & TypeSpec.Surfref) == TypeSpec.Surfref; }

        public static bool is_scalar(this Type type) { return !type.is_opaque() && !type.is_vec() && !type.is_arr(); }
        public static bool is_int(this Type type) { return (type.spec() & TypeSpec.Integer) == TypeSpec.Integer; }
        public static bool is_signed(this Type type) { return (type.spec() & TypeSpec.Signed) == TypeSpec.Signed; }
        public static bool is_unsigned(this Type type) { return (type.spec() & TypeSpec.Unsigned) == TypeSpec.Unsigned; }
        public static bool is_float(this Type type) { return (type.spec() & TypeSpec.Float) == TypeSpec.Float; }
        public static bool is_bit(this Type type) { return (type.spec() & TypeSpec.Bit) == TypeSpec.Bit; }

        public static bool is_vec(this Type type) { return type.vec_rank() != 0; }
        public static bool is_v1(this Type type) { return type.vec_rank() == 1; }
        public static bool is_v2(this Type type) { return type.vec_rank() == 2; }
        public static bool is_v4(this Type type) { return type.vec_rank() == 4; }
        public static int vec_rank(this Type type)
        {
            if (type == null) return 0;
            if (type.is_arr()) return 0;
            if ((type.Mod & TypeMod.V1) == TypeMod.V1) return 1;
            if ((type.Mod & TypeMod.V2) == TypeMod.V2) return 2;
            if ((type.Mod & TypeMod.V4) == TypeMod.V4) return 4;
            return 0;
        }

        public static Type vec_el(this Type type)
        {
            if (type == null) return null;
            if (!type.is_vec()) return null;

            var el = new Type();
            el.Name = type.Name;
            el.Mod = type.Mod & ~(TypeMod.V1 | TypeMod.V2 | TypeMod.V4);
            el.Dims = new int[0];
            return el;
        }

        public static bool is_arr(this Type type) { return type.arr_rank() != 0; }
        public static int arr_rank(this Type type)
        {
            if (type == null) return 0;
            return type.Dims == null ? 0 : type.Dims.Length;
        }

        public static Type arr_el(this Type type)
        {
            if (type == null) return null;
            if (!type.is_arr()) return null;

            var el = new Type();
            el.Name = type.Name;
            el.Mod = type.Mod;
            el.Dims = (type.Dims ?? new int[0]).SkipLast(1).ToArray();
            if (el.Dims.Count() == 0) el.Mod &= ~TypeMod.Array;
            return el;
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
                    return 0;
                case TypeName.Texref:
                    return 0;
                case TypeName.Samplerref:
                    return 0;
                case TypeName.Surfref:
                    return 0;
                default:
                    throw AssertionHelper.Fail();
            }
        }
    }
}