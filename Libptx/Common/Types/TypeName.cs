using System.Collections.Generic;
using System.Diagnostics;
using Libcuda.DataTypes;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types.Bits;
using Libptx.Common.Types.Opaques;
using Libptx.Common.Types.Pointers;
using XenoGears.Functional;
using ClrType = System.Type;

namespace Libptx.Common.Types
{
    public enum TypeName
    {
        [Affix("u8")] U8 = 1,
        [Affix("s8")] S8,
        [Affix("u16")] U16,
        [Affix("s16")] S16,
        [Affix("u32")] U32,
        [Affix("s32")] S32,
        [Affix("u64")] U64,
        [Affix("s64")] S64,
        [Affix("f16")] F16,
        [Affix("f32")] F32,
        [Affix("f64")] F64,
        [Affix("b8")] B8,
        [Affix("b16")] B16,
        [Affix("b32")] B32,
        [Affix("b64")] B64,
        [Affix("pred")] Pred,
        [Affix("texref")] Texref,
        [Affix15("samplerref")] Samplerref,
        [Affix15("surfref")] Surfref,
        [Affix("ptr")] Ptr,
        [Affix("bmk")] Bmk,
    }

    [DebuggerNonUserCode]
    public static class TypeNameExtensions
    {
        private static readonly Dictionary<TypeName, ClrType> pool = new Dictionary<TypeName, ClrType>();

        static TypeNameExtensions()
        {
            pool.Add(TypeName.U8, typeof(byte));
            pool.Add(TypeName.S8, typeof(sbyte));
            pool.Add(TypeName.U16, typeof(ushort));
            pool.Add(TypeName.S16, typeof(short));
            pool.Add(TypeName.U32, typeof(uint));
            pool.Add(TypeName.S32, typeof(int));
            pool.Add(TypeName.U64, typeof(ulong));
            pool.Add(TypeName.S64, typeof(long));
            pool.Add(TypeName.F16, typeof(half));
            pool.Add(TypeName.F32, typeof(float));
            pool.Add(TypeName.F64, typeof(double));
            pool.Add(TypeName.B8, typeof(Bit8));
            pool.Add(TypeName.B16, typeof(Bit16));
            pool.Add(TypeName.B32, typeof(Bit32));
            pool.Add(TypeName.B64, typeof(Bit64));
            pool.Add(TypeName.Pred, typeof(bool));
            pool.Add(TypeName.Texref, typeof(Texref));
            pool.Add(TypeName.Samplerref, typeof(Samplerref));
            pool.Add(TypeName.Surfref, typeof(Surfref));
            pool.Add(TypeName.Ptr, typeof(Ptr));
            pool.Add(TypeName.Bmk, typeof(Bmk));
        }

        public static ClrType ClrType(this TypeName t)
        {
            return pool.GetOrDefault(t);
        }
    }
}