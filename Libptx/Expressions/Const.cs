using System;
using System.IO;
using Libcuda.DataTypes;
using Libptx.Common;
using XenoGears.Assertions;
using XenoGears.Functional;
using System.Linq;

namespace Libptx.Expressions
{
    public class Const : Atom, Expression
    {
        public static implicit operator Const(bool value) { return new Const(value); }
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
        public static implicit operator Const(half value) { return new Const(value); }
        public static implicit operator Const(half1 value) { return new Const(value); }
        public static implicit operator Const(half2 value) { return new Const(value); }
        public static implicit operator Const(half3 value) { return new Const(value); }
        public static implicit operator Const(half4 value) { return new Const(value); }
        public static implicit operator Const(float value) { return new Const(value); }
        public static implicit operator Const(float1 value) { return new Const(value); }
        public static implicit operator Const(float2 value) { return new Const(value); }
        public static implicit operator Const(float3 value) { return new Const(value); }
        public static implicit operator Const(float4 value) { return new Const(value); }
        public static implicit operator Const(double value) { return new Const(value); }
        public static implicit operator Const(double1 value) { return new Const(value); }
        public static implicit operator Const(double2 value) { return new Const(value); }

        // todo. also textgen strongly-typed implicit casts from T[], T[,] and T[][] where T is one of supported types
        public static implicit operator Const(Array value) { return new Const(value); }

        public Const(Object value)
        {
            Value = value;
        }

        private Object _value;
        public Object Value
        {
            get { return _value; }
            set
            {
                ValidateValue(value);
                _value = value;
            }
        }

        private void ValidateValue(Object value)
        {
            value.AssertNotNull();

            var elt = value.GetType().Unfold(t => t.IsArray ? t.GetElementType() : null, t => t != null).Last();
            (elt.IsCudaPrimitive() || elt.IsCudaVector()).AssertTrue();
        }

        protected override void CustomValidate(Module ctx)
        {
            ValidateValue(Value);
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}