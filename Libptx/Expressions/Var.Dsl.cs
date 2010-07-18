using System;
using Libcuda.DataTypes;
using Libptx.Common.Types;
using XenoGears.Assertions;
using Type = Libptx.Common.Types.Type;
using ClrType = System.Type;

namespace Libptx.Expressions
{
    public partial class Var
    {
        public static implicit operator Var(Type t) { return new Var { Type = t }; }
        public static implicit operator Var(ClrType t) { return new Var { Type = t }; }
        public static implicit operator Var(TypeName t) { return new Var { Type = t }; }

        public new Var_U8 u8 { get { return Clone(v => v.Type = TypeName.U8, v => v.Init = null); } }
        public new Var_S8 s8 { get { return Clone(v => v.Type = TypeName.S8, v => v.Init = null); } }
        public new Var_U16 u16 { get { return Clone(v => v.Type = TypeName.U16, v => v.Init = null); } }
        public new Var_S16 s16 { get { return Clone(v => v.Type = TypeName.S16, v => v.Init = null); } }
        public new Var_U24 u24 { get { return Clone(v => v.Type = TypeName.U24, v => v.Init = null); } }
        public new Var_S24 s24 { get { return Clone(v => v.Type = TypeName.S24, v => v.Init = null); } }
        public new Var_U32 u32 { get { return Clone(v => v.Type = TypeName.U32, v => v.Init = null); } }
        public new Var_S32 s32 { get { return Clone(v => v.Type = TypeName.S32, v => v.Init = null); } }
        public new Var_U64 u64 { get { return Clone(v => v.Type = TypeName.U64, v => v.Init = null); } }
        public new Var_S64 s64 { get { return Clone(v => v.Type = TypeName.S64, v => v.Init = null); } }
        public new Var_F16 f16 { get { return Clone(v => v.Type = TypeName.F16, v => v.Init = null); } }
        public new Var_F32 f32 { get { return Clone(v => v.Type = TypeName.F32, v => v.Init = null); } }
        public new Var_F64 f64 { get { return Clone(v => v.Type = TypeName.F64, v => v.Init = null); } }
        public new Var_B8 b8 { get { return Clone(v => v.Type = TypeName.B8, v => v.Init = null); } }
        public new Var_B16 b16 { get { return Clone(v => v.Type = TypeName.B16, v => v.Init = null); } }
        public new Var_B32 b32 { get { return Clone(v => v.Type = TypeName.B32, v => v.Init = null); } }
        public new Var_B64 b64 { get { return Clone(v => v.Type = TypeName.B64, v => v.Init = null); } }
        public new Var_Pred pred { get { return Clone(v => v.Type = TypeName.Pred, v => v.Init = null); } }
        public new Var_Tex tex { get { return Clone(v => v.Type = TypeName.Tex, v => v.Init = null); } }
        public new Var_Sampler sampler { get { return Clone(v => v.Type = TypeName.Sampler, v => v.Init = null); } }
        public new Var_Surf surf { get { return Clone(v => v.Type = TypeName.Surf, v => v.Init = null); } }
        protected Var v1 { get { return Clone(v => v.Type = v.Type.v1, v => v.Init = null); } }
        protected Var v2 { get { return Clone(v => v.Type = v.Type.v2, v => v.Init = null); } }
        protected Var v4 { get { return Clone(v => v.Type = v.Type.v4, v => v.Init = null); } }
        protected Var this[params int[] dims] { get { return Clone(v => v.Type = v.Type[dims], v => v.Init = null); } }

        protected new Var reg { get { return Clone(v => v.Space = Common.Enumerations.Space.Register); } }
        protected new Var sreg { get { return Clone(v => v.Space = Common.Enumerations.Space.Special); } }
        protected new Var local { get { return Clone(v => v.Space = Common.Enumerations.Space.Local); } }
        protected new Var shared { get { return Clone(v => v.Space = Common.Enumerations.Space.Shared); } }
        protected new Var global { get { return Clone(v => v.Space = Common.Enumerations.Space.Global); } }
        protected new Var param { get { return Clone(v => v.Space = Common.Enumerations.Space.Param); } }
        protected new Var @const { get { return Clone(v => v.Space = Common.Enumerations.Space.Const); } }
        protected new Var const0 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const0); } }
        protected new Var const1 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const1); } }
        protected new Var const2 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const2); } }
        protected new Var const3 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const3); } }
        protected new Var const4 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const4); } }
        protected new Var const5 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const5); } }
        protected new Var const6 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const6); } }
        protected new Var const7 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const7); } }
        protected new Var const8 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const8); } }
        protected new Var const9 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const9); } }
        protected new Var const10 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const10); } }

        protected Var init(bool value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte1 value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte2 value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte3 value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte4 value) { return Clone(v => v.Init = value); }
        protected Var init(byte value) { return Clone(v => v.Init = value); }
        protected Var init(byte1 value) { return Clone(v => v.Init = value); }
        protected Var init(byte2 value) { return Clone(v => v.Init = value); }
        protected Var init(byte3 value) { return Clone(v => v.Init = value); }
        protected Var init(byte4 value) { return Clone(v => v.Init = value); }
        protected Var init(short value) { return Clone(v => v.Init = value); }
        protected Var init(short1 value) { return Clone(v => v.Init = value); }
        protected Var init(short2 value) { return Clone(v => v.Init = value); }
        protected Var init(short3 value) { return Clone(v => v.Init = value); }
        protected Var init(short4 value) { return Clone(v => v.Init = value); }
        protected Var init(ushort value) { return Clone(v => v.Init = value); }
        protected Var init(ushort1 value) { return Clone(v => v.Init = value); }
        protected Var init(ushort2 value) { return Clone(v => v.Init = value); }
        protected Var init(ushort3 value) { return Clone(v => v.Init = value); }
        protected Var init(ushort4 value) { return Clone(v => v.Init = value); }
        protected Var init(int value) { return Clone(v => v.Init = value); }
        protected Var init(int1 value) { return Clone(v => v.Init = value); }
        protected Var init(int2 value) { return Clone(v => v.Init = value); }
        protected Var init(int3 value) { return Clone(v => v.Init = value); }
        protected Var init(int4 value) { return Clone(v => v.Init = value); }
        protected Var init(uint value) { return Clone(v => v.Init = value); }
        protected Var init(uint1 value) { return Clone(v => v.Init = value); }
        protected Var init(uint2 value) { return Clone(v => v.Init = value); }
        protected Var init(uint3 value) { return Clone(v => v.Init = value); }
        protected Var init(uint4 value) { return Clone(v => v.Init = value); }
        protected Var init(long value) { return Clone(v => v.Init = value); }
        protected Var init(long1 value) { return Clone(v => v.Init = value); }
        protected Var init(long2 value) { return Clone(v => v.Init = value); }
        protected Var init(ulong value) { return Clone(v => v.Init = value); }
        protected Var init(ulong1 value) { return Clone(v => v.Init = value); }
        protected Var init(ulong2 value) { return Clone(v => v.Init = value); }
        protected Var init(half value) { return Clone(v => v.Init = value); }
        protected Var init(half1 value) { return Clone(v => v.Init = value); }
        protected Var init(half2 value) { return Clone(v => v.Init = value); }
        protected Var init(half3 value) { return Clone(v => v.Init = value); }
        protected Var init(half4 value) { return Clone(v => v.Init = value); }
        protected Var init(float value) { return Clone(v => v.Init = value); }
        protected Var init(float1 value) { return Clone(v => v.Init = value); }
        protected Var init(float2 value) { return Clone(v => v.Init = value); }
        protected Var init(float3 value) { return Clone(v => v.Init = value); }
        protected Var init(float4 value) { return Clone(v => v.Init = value); }
        protected Var init(double value) { return Clone(v => v.Init = value); }
        protected Var init(double1 value) { return Clone(v => v.Init = value); }
        protected Var init(double2 value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte[] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte[][] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte[,] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte1[] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte1[][] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte1[,] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte1[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte1[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte2[] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte2[][] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte2[,] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte2[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte2[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte3[] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte3[][] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte3[,] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte3[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte3[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte4[] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte4[][] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte4[,] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte4[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(sbyte4[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(byte[] value) { return Clone(v => v.Init = value); }
        protected Var init(byte[][] value) { return Clone(v => v.Init = value); }
        protected Var init(byte[,] value) { return Clone(v => v.Init = value); }
        protected Var init(byte[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(byte[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(byte1[] value) { return Clone(v => v.Init = value); }
        protected Var init(byte1[][] value) { return Clone(v => v.Init = value); }
        protected Var init(byte1[,] value) { return Clone(v => v.Init = value); }
        protected Var init(byte1[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(byte1[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(byte2[] value) { return Clone(v => v.Init = value); }
        protected Var init(byte2[][] value) { return Clone(v => v.Init = value); }
        protected Var init(byte2[,] value) { return Clone(v => v.Init = value); }
        protected Var init(byte2[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(byte2[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(byte3[] value) { return Clone(v => v.Init = value); }
        protected Var init(byte3[][] value) { return Clone(v => v.Init = value); }
        protected Var init(byte3[,] value) { return Clone(v => v.Init = value); }
        protected Var init(byte3[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(byte3[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(byte4[] value) { return Clone(v => v.Init = value); }
        protected Var init(byte4[][] value) { return Clone(v => v.Init = value); }
        protected Var init(byte4[,] value) { return Clone(v => v.Init = value); }
        protected Var init(byte4[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(byte4[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(short[] value) { return Clone(v => v.Init = value); }
        protected Var init(short[][] value) { return Clone(v => v.Init = value); }
        protected Var init(short[,] value) { return Clone(v => v.Init = value); }
        protected Var init(short[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(short[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(short1[] value) { return Clone(v => v.Init = value); }
        protected Var init(short1[][] value) { return Clone(v => v.Init = value); }
        protected Var init(short1[,] value) { return Clone(v => v.Init = value); }
        protected Var init(short1[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(short1[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(short2[] value) { return Clone(v => v.Init = value); }
        protected Var init(short2[][] value) { return Clone(v => v.Init = value); }
        protected Var init(short2[,] value) { return Clone(v => v.Init = value); }
        protected Var init(short2[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(short2[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(short3[] value) { return Clone(v => v.Init = value); }
        protected Var init(short3[][] value) { return Clone(v => v.Init = value); }
        protected Var init(short3[,] value) { return Clone(v => v.Init = value); }
        protected Var init(short3[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(short3[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(short4[] value) { return Clone(v => v.Init = value); }
        protected Var init(short4[][] value) { return Clone(v => v.Init = value); }
        protected Var init(short4[,] value) { return Clone(v => v.Init = value); }
        protected Var init(short4[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(short4[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort[] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort[][] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort[,] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort1[] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort1[][] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort1[,] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort1[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort1[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort2[] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort2[][] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort2[,] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort2[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort2[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort3[] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort3[][] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort3[,] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort3[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort3[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort4[] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort4[][] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort4[,] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort4[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(ushort4[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(int[] value) { return Clone(v => v.Init = value); }
        protected Var init(int[][] value) { return Clone(v => v.Init = value); }
        protected Var init(int[,] value) { return Clone(v => v.Init = value); }
        protected Var init(int[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(int[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(int1[] value) { return Clone(v => v.Init = value); }
        protected Var init(int1[][] value) { return Clone(v => v.Init = value); }
        protected Var init(int1[,] value) { return Clone(v => v.Init = value); }
        protected Var init(int1[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(int1[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(int2[] value) { return Clone(v => v.Init = value); }
        protected Var init(int2[][] value) { return Clone(v => v.Init = value); }
        protected Var init(int2[,] value) { return Clone(v => v.Init = value); }
        protected Var init(int2[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(int2[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(int3[] value) { return Clone(v => v.Init = value); }
        protected Var init(int3[][] value) { return Clone(v => v.Init = value); }
        protected Var init(int3[,] value) { return Clone(v => v.Init = value); }
        protected Var init(int3[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(int3[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(int4[] value) { return Clone(v => v.Init = value); }
        protected Var init(int4[][] value) { return Clone(v => v.Init = value); }
        protected Var init(int4[,] value) { return Clone(v => v.Init = value); }
        protected Var init(int4[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(int4[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(uint[] value) { return Clone(v => v.Init = value); }
        protected Var init(uint[][] value) { return Clone(v => v.Init = value); }
        protected Var init(uint[,] value) { return Clone(v => v.Init = value); }
        protected Var init(uint[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(uint[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(uint1[] value) { return Clone(v => v.Init = value); }
        protected Var init(uint1[][] value) { return Clone(v => v.Init = value); }
        protected Var init(uint1[,] value) { return Clone(v => v.Init = value); }
        protected Var init(uint1[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(uint1[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(uint2[] value) { return Clone(v => v.Init = value); }
        protected Var init(uint2[][] value) { return Clone(v => v.Init = value); }
        protected Var init(uint2[,] value) { return Clone(v => v.Init = value); }
        protected Var init(uint2[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(uint2[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(uint3[] value) { return Clone(v => v.Init = value); }
        protected Var init(uint3[][] value) { return Clone(v => v.Init = value); }
        protected Var init(uint3[,] value) { return Clone(v => v.Init = value); }
        protected Var init(uint3[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(uint3[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(uint4[] value) { return Clone(v => v.Init = value); }
        protected Var init(uint4[][] value) { return Clone(v => v.Init = value); }
        protected Var init(uint4[,] value) { return Clone(v => v.Init = value); }
        protected Var init(uint4[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(uint4[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(long[] value) { return Clone(v => v.Init = value); }
        protected Var init(long[][] value) { return Clone(v => v.Init = value); }
        protected Var init(long[,] value) { return Clone(v => v.Init = value); }
        protected Var init(long[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(long[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(long1[] value) { return Clone(v => v.Init = value); }
        protected Var init(long1[][] value) { return Clone(v => v.Init = value); }
        protected Var init(long1[,] value) { return Clone(v => v.Init = value); }
        protected Var init(long1[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(long1[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(long2[] value) { return Clone(v => v.Init = value); }
        protected Var init(long2[][] value) { return Clone(v => v.Init = value); }
        protected Var init(long2[,] value) { return Clone(v => v.Init = value); }
        protected Var init(long2[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(long2[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(ulong[] value) { return Clone(v => v.Init = value); }
        protected Var init(ulong[][] value) { return Clone(v => v.Init = value); }
        protected Var init(ulong[,] value) { return Clone(v => v.Init = value); }
        protected Var init(ulong[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(ulong[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(ulong1[] value) { return Clone(v => v.Init = value); }
        protected Var init(ulong1[][] value) { return Clone(v => v.Init = value); }
        protected Var init(ulong1[,] value) { return Clone(v => v.Init = value); }
        protected Var init(ulong1[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(ulong1[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(ulong2[] value) { return Clone(v => v.Init = value); }
        protected Var init(ulong2[][] value) { return Clone(v => v.Init = value); }
        protected Var init(ulong2[,] value) { return Clone(v => v.Init = value); }
        protected Var init(ulong2[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(ulong2[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(half[] value) { return Clone(v => v.Init = value); }
        protected Var init(half[][] value) { return Clone(v => v.Init = value); }
        protected Var init(half[,] value) { return Clone(v => v.Init = value); }
        protected Var init(half[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(half[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(half1[] value) { return Clone(v => v.Init = value); }
        protected Var init(half1[][] value) { return Clone(v => v.Init = value); }
        protected Var init(half1[,] value) { return Clone(v => v.Init = value); }
        protected Var init(half1[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(half1[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(half2[] value) { return Clone(v => v.Init = value); }
        protected Var init(half2[][] value) { return Clone(v => v.Init = value); }
        protected Var init(half2[,] value) { return Clone(v => v.Init = value); }
        protected Var init(half2[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(half2[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(half3[] value) { return Clone(v => v.Init = value); }
        protected Var init(half3[][] value) { return Clone(v => v.Init = value); }
        protected Var init(half3[,] value) { return Clone(v => v.Init = value); }
        protected Var init(half3[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(half3[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(half4[] value) { return Clone(v => v.Init = value); }
        protected Var init(half4[][] value) { return Clone(v => v.Init = value); }
        protected Var init(half4[,] value) { return Clone(v => v.Init = value); }
        protected Var init(half4[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(half4[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(float[] value) { return Clone(v => v.Init = value); }
        protected Var init(float[][] value) { return Clone(v => v.Init = value); }
        protected Var init(float[,] value) { return Clone(v => v.Init = value); }
        protected Var init(float[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(float[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(float1[] value) { return Clone(v => v.Init = value); }
        protected Var init(float1[][] value) { return Clone(v => v.Init = value); }
        protected Var init(float1[,] value) { return Clone(v => v.Init = value); }
        protected Var init(float1[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(float1[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(float2[] value) { return Clone(v => v.Init = value); }
        protected Var init(float2[][] value) { return Clone(v => v.Init = value); }
        protected Var init(float2[,] value) { return Clone(v => v.Init = value); }
        protected Var init(float2[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(float2[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(float3[] value) { return Clone(v => v.Init = value); }
        protected Var init(float3[][] value) { return Clone(v => v.Init = value); }
        protected Var init(float3[,] value) { return Clone(v => v.Init = value); }
        protected Var init(float3[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(float3[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(float4[] value) { return Clone(v => v.Init = value); }
        protected Var init(float4[][] value) { return Clone(v => v.Init = value); }
        protected Var init(float4[,] value) { return Clone(v => v.Init = value); }
        protected Var init(float4[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(float4[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(double[] value) { return Clone(v => v.Init = value); }
        protected Var init(double[][] value) { return Clone(v => v.Init = value); }
        protected Var init(double[,] value) { return Clone(v => v.Init = value); }
        protected Var init(double[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(double[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(double1[] value) { return Clone(v => v.Init = value); }
        protected Var init(double1[][] value) { return Clone(v => v.Init = value); }
        protected Var init(double1[,] value) { return Clone(v => v.Init = value); }
        protected Var init(double1[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(double1[, ,] value) { return Clone(v => v.Init = value); }
        protected Var init(double2[] value) { return Clone(v => v.Init = value); }
        protected Var init(double2[][] value) { return Clone(v => v.Init = value); }
        protected Var init(double2[,] value) { return Clone(v => v.Init = value); }
        protected Var init(double2[][][] value) { return Clone(v => v.Init = value); }
        protected Var init(double2[, ,] value) { return Clone(v => v.Init = value); }

        protected Var align(int alignment) { return Clone(v => v.Alignment = alignment.AssertThat(i => i > 0)); }
        protected Var align1 { get { return align(1); } }
        protected Var align2 { get { return align(2); } }
        protected Var align4 { get { return align(4); } }
        protected Var align8 { get { return align(8); } }
        protected Var align16 { get { return align(16); } }

        // protected static Var operator !(Var @var) { return @var.Clone(v => v.Mod = VarMod.Not); }
        // protected static Var operator -(Var @var) { return @var.Clone(v => v.Mod = VarMod.Neg); }
        // protected static Couple operator |(Var fst, Var snd) { return new Couple{Fst = fst, Snd = snd}; }
        protected Var b0 { get { return Clone(v => v.Mod = VarMod.B0); } }
        protected Var b1 { get { return Clone(v => v.Mod = VarMod.B1); } }
        protected Var b2 { get { return Clone(v => v.Mod = VarMod.B2); } }
        protected Var b3 { get { return Clone(v => v.Mod = VarMod.B3); } }
        protected Var h0 { get { return Clone(v => v.Mod = VarMod.H0); } }
        protected Var h1 { get { return Clone(v => v.Mod = VarMod.H1); } }
        protected Var x { get { return Clone(v => v.Mod = VarMod.X); } }
        protected Var r { get { return Clone(v => v.Mod = VarMod.R); } }
        protected Var y { get { return Clone(v => v.Mod = VarMod.Y); } }
        protected Var g { get { return Clone(v => v.Mod = VarMod.G); } }
        protected Var z { get { return Clone(v => v.Mod = VarMod.Z); } }
        protected Var b { get { return Clone(v => v.Mod = VarMod.B); } }
        protected Var w { get { return Clone(v => v.Mod = VarMod.W); } }
        protected Var a { get { return Clone(v => v.Mod = VarMod.A); } }

        protected Var export { get { return Clone(v => v.IsVisible = true); } }
        protected Var import { get { return Clone(v => v.IsExtern = true); } }
        protected Var @extern { get { return Clone(v => v.IsExtern = true); } }

        private Var Clone()
        {
            var clone = new Var();
            clone.Name = this.Name;
            clone.Space = this.Space;
            clone.Type = this.Type;
            clone.Init = this.Init;
            clone.Alignment = this.Alignment;
            clone.Mod = this.Mod;
            clone.IsVisible = this.IsVisible;
            clone.IsExtern = this.IsExtern;
            return clone;
        }

        private Var Clone(params Action<Var>[] mods)
        {
            var clone = Clone();
            foreach (var mod in mods) mod(clone);
            return clone;
        }
    }
}
