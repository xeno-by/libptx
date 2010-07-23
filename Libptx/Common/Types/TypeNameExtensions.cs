using System.Collections.Generic;
using Libcuda.DataTypes;
using XenoGears.Functional;
using ClrType = System.Type;

namespace Libptx.Common.Types
{
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
        }

        public static ClrType ClrType(this TypeName t)
        {
            return pool.GetOrDefault(t);
        }
    }
}