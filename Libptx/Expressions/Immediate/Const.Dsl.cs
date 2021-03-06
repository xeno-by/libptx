﻿using Libcuda.DataTypes;
using Libptx.Common.Types.Bits;
using Libptx.Common.Types.Opaques;
using Libptx.Expressions.Addresses;

namespace Libptx.Expressions.Immediate
{
    public partial class Const
    {
        // predicates
        public static implicit operator Const(bool value) { return new Const(value); }

        // opaques
        public static implicit operator Const(Texref value) { return new Const(value); }
        public static implicit operator Const(Samplerref value) { return new Const(value); }
        public static implicit operator Const(Surfref value) { return new Const(value); }

        // addresses
        public static implicit operator Const(Address value) { return new Const(value); }

        // scalars and their derivatives
        public static implicit operator Const(sbyte value) { return new Const(value); }
        public static implicit operator Const(sbyte1 value) { return new Const(value); }
        public static implicit operator Const(sbyte2 value) { return new Const(value); }
        public static implicit operator Const(sbyte3 value) { return new Const(value); }
        public static implicit operator Const(sbyte4 value) { return new Const(value); }
        public static implicit operator Const(byte value) { return new Const(value); }
        public static implicit operator Const(byte1 value) { return new Const(value); }
        public static implicit operator Const(byte2 value) { return new Const(value); }
        public static implicit operator Const(byte3 value) { return new Const(value); }
        public static implicit operator Const(byte4 value) { return new Const(value); }
        public static implicit operator Const(short value) { return new Const(value); }
        public static implicit operator Const(short1 value) { return new Const(value); }
        public static implicit operator Const(short2 value) { return new Const(value); }
        public static implicit operator Const(short3 value) { return new Const(value); }
        public static implicit operator Const(short4 value) { return new Const(value); }
        public static implicit operator Const(ushort value) { return new Const(value); }
        public static implicit operator Const(ushort1 value) { return new Const(value); }
        public static implicit operator Const(ushort2 value) { return new Const(value); }
        public static implicit operator Const(ushort3 value) { return new Const(value); }
        public static implicit operator Const(ushort4 value) { return new Const(value); }
        public static implicit operator Const(int value) { return new Const(value); }
        public static implicit operator Const(int1 value) { return new Const(value); }
        public static implicit operator Const(int2 value) { return new Const(value); }
        public static implicit operator Const(int3 value) { return new Const(value); }
        public static implicit operator Const(int4 value) { return new Const(value); }
        public static implicit operator Const(uint value) { return new Const(value); }
        public static implicit operator Const(uint1 value) { return new Const(value); }
        public static implicit operator Const(uint2 value) { return new Const(value); }
        public static implicit operator Const(uint3 value) { return new Const(value); }
        public static implicit operator Const(uint4 value) { return new Const(value); }
        public static implicit operator Const(long value) { return new Const(value); }
        public static implicit operator Const(long1 value) { return new Const(value); }
        public static implicit operator Const(long2 value) { return new Const(value); }
        public static implicit operator Const(ulong value) { return new Const(value); }
        public static implicit operator Const(ulong1 value) { return new Const(value); }
        public static implicit operator Const(ulong2 value) { return new Const(value); }
        public static implicit operator Const(float value) { return new Const(value); }
        public static implicit operator Const(float1 value) { return new Const(value); }
        public static implicit operator Const(float2 value) { return new Const(value); }
        public static implicit operator Const(float3 value) { return new Const(value); }
        public static implicit operator Const(float4 value) { return new Const(value); }
        public static implicit operator Const(double value) { return new Const(value); }
        public static implicit operator Const(double1 value) { return new Const(value); }
        public static implicit operator Const(double2 value) { return new Const(value); }
        public static implicit operator Const(Bit8 value) { return new Const(value); }
        public static implicit operator Const(Bit8_V1 value) { return new Const(value); }
        public static implicit operator Const(Bit8_V2 value) { return new Const(value); }
        public static implicit operator Const(Bit8_V3 value) { return new Const(value); }
        public static implicit operator Const(Bit8_V4 value) { return new Const(value); }
        public static implicit operator Const(Bit16 value) { return new Const(value); }
        public static implicit operator Const(Bit16_V1 value) { return new Const(value); }
        public static implicit operator Const(Bit16_V2 value) { return new Const(value); }
        public static implicit operator Const(Bit16_V3 value) { return new Const(value); }
        public static implicit operator Const(Bit16_V4 value) { return new Const(value); }
        public static implicit operator Const(Bit32 value) { return new Const(value); }
        public static implicit operator Const(Bit32_V1 value) { return new Const(value); }
        public static implicit operator Const(Bit32_V2 value) { return new Const(value); }
        public static implicit operator Const(Bit32_V3 value) { return new Const(value); }
        public static implicit operator Const(Bit32_V4 value) { return new Const(value); }
        public static implicit operator Const(Bit64 value) { return new Const(value); }
        public static implicit operator Const(Bit64_V1 value) { return new Const(value); }
        public static implicit operator Const(Bit64_V2 value) { return new Const(value); }
        public static implicit operator Const(sbyte[] value) { return new Const(value); }
        public static implicit operator Const(sbyte1[] value) { return new Const(value); }
        public static implicit operator Const(sbyte2[] value) { return new Const(value); }
        public static implicit operator Const(sbyte3[] value) { return new Const(value); }
        public static implicit operator Const(sbyte4[] value) { return new Const(value); }
        public static implicit operator Const(byte[] value) { return new Const(value); }
        public static implicit operator Const(byte1[] value) { return new Const(value); }
        public static implicit operator Const(byte2[] value) { return new Const(value); }
        public static implicit operator Const(byte3[] value) { return new Const(value); }
        public static implicit operator Const(byte4[] value) { return new Const(value); }
        public static implicit operator Const(short[] value) { return new Const(value); }
        public static implicit operator Const(short1[] value) { return new Const(value); }
        public static implicit operator Const(short2[] value) { return new Const(value); }
        public static implicit operator Const(short3[] value) { return new Const(value); }
        public static implicit operator Const(short4[] value) { return new Const(value); }
        public static implicit operator Const(ushort[] value) { return new Const(value); }
        public static implicit operator Const(ushort1[] value) { return new Const(value); }
        public static implicit operator Const(ushort2[] value) { return new Const(value); }
        public static implicit operator Const(ushort3[] value) { return new Const(value); }
        public static implicit operator Const(ushort4[] value) { return new Const(value); }
        public static implicit operator Const(int[] value) { return new Const(value); }
        public static implicit operator Const(int1[] value) { return new Const(value); }
        public static implicit operator Const(int2[] value) { return new Const(value); }
        public static implicit operator Const(int3[] value) { return new Const(value); }
        public static implicit operator Const(int4[] value) { return new Const(value); }
        public static implicit operator Const(uint[] value) { return new Const(value); }
        public static implicit operator Const(uint1[] value) { return new Const(value); }
        public static implicit operator Const(uint2[] value) { return new Const(value); }
        public static implicit operator Const(uint3[] value) { return new Const(value); }
        public static implicit operator Const(uint4[] value) { return new Const(value); }
        public static implicit operator Const(long[] value) { return new Const(value); }
        public static implicit operator Const(long1[] value) { return new Const(value); }
        public static implicit operator Const(long2[] value) { return new Const(value); }
        public static implicit operator Const(ulong[] value) { return new Const(value); }
        public static implicit operator Const(ulong1[] value) { return new Const(value); }
        public static implicit operator Const(ulong2[] value) { return new Const(value); }
        public static implicit operator Const(float[] value) { return new Const(value); }
        public static implicit operator Const(float1[] value) { return new Const(value); }
        public static implicit operator Const(float2[] value) { return new Const(value); }
        public static implicit operator Const(float3[] value) { return new Const(value); }
        public static implicit operator Const(float4[] value) { return new Const(value); }
        public static implicit operator Const(double[] value) { return new Const(value); }
        public static implicit operator Const(double1[] value) { return new Const(value); }
        public static implicit operator Const(double2[] value) { return new Const(value); }
        public static implicit operator Const(Bit8[] value) { return new Const(value); }
        public static implicit operator Const(Bit8_V1[] value) { return new Const(value); }
        public static implicit operator Const(Bit8_V2[] value) { return new Const(value); }
        public static implicit operator Const(Bit8_V3[] value) { return new Const(value); }
        public static implicit operator Const(Bit8_V4[] value) { return new Const(value); }
        public static implicit operator Const(Bit16[] value) { return new Const(value); }
        public static implicit operator Const(Bit16_V1[] value) { return new Const(value); }
        public static implicit operator Const(Bit16_V2[] value) { return new Const(value); }
        public static implicit operator Const(Bit16_V3[] value) { return new Const(value); }
        public static implicit operator Const(Bit16_V4[] value) { return new Const(value); }
        public static implicit operator Const(Bit32[] value) { return new Const(value); }
        public static implicit operator Const(Bit32_V1[] value) { return new Const(value); }
        public static implicit operator Const(Bit32_V2[] value) { return new Const(value); }
        public static implicit operator Const(Bit32_V3[] value) { return new Const(value); }
        public static implicit operator Const(Bit32_V4[] value) { return new Const(value); }
        public static implicit operator Const(Bit64[] value) { return new Const(value); }
        public static implicit operator Const(Bit64_V1[] value) { return new Const(value); }
        public static implicit operator Const(Bit64_V2[] value) { return new Const(value); }
    }
}
