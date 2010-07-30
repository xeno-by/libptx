using System;
using System.Collections;
using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Common.Types;
using Libptx.Common.Types.Bits;
using Libptx.Common.Types.Opaques;
using Libptx.Expressions.Addresses;
using Libptx.Reflection;
using XenoGears.Assertions;
using XenoGears.Functional;
using Type=Libptx.Common.Types.Type;
using XenoGears.Strings;

namespace Libptx.Expressions.Immediate
{
    [DebuggerNonUserCode]
    public partial class Const : Atom, Expression
    {
        public Const() {}
        public Const(Object value) { Value = value; }

        public Object Value { get; set; }
        public Type Type { get { return Value == null ? null : Value.GetType(); } }

        protected override SoftwareIsa CustomVersion { get { return Value.Version(); } }
        protected override HardwareIsa CustomTarget { get { return Value.Target(); } }

        protected override void CustomValidate()
        {
            (Type != null).AssertTrue();
            Type.Validate();

            (Value != null).AssertTrue();
            if (Value is Address)
            {
                var v_addr = Value as Address;
                (v_addr.Base == null).AssertTrue();
            }
            else
            {
                var t = (Type)Value.GetType();
                (t != null).AssertTrue();
                (t.el() == f16 || t.el().is_ptr() || t.el().is_bmk()).AssertFalse();
            }
        }

        protected override void RenderPtx()
        {
            // predicates
            if (Value is bool)
            {
                var v_bool = (bool)Value;
                writer.Write(v_bool ? "1" : "0");
            }
            // opaques
            else if (Value is Texref || Value is Samplerref || Value is Surfref)
            {
                Value.Quanta().ForEach((kvp, i) =>
                {
                    if (i != 0) writer.Write(", ");

                    var k = kvp.Key;
                    writer.Write(k + " = ");

                    var v = kvp.Value;
                    var s_v = v.Signature() ?? v.ToInvariantString();
                    writer.Write(s_v);
                });
            }
            // addresses
            else if (Value is Address)
            {
                var v_addr = (Address)Value;
                var s_addr = v_addr.RunRenderPtx();
                if (s_addr.StartsWith("[")) s_addr = s_addr.Slice(1);
                if (s_addr.EndsWith("]")) s_addr = s_addr.Slice(-1);
                writer.Write(s_addr);
            }
            // scalars and their derivatives
            else
            {
                Action<Object> render_scalar = v =>
                {
                    v.AssertNotNull();
                    if (v is sbyte)
                    {
                        var v_sbyte = (sbyte)v;
                        writer.Write("{0} /* {1} */", v_sbyte.ToString("x2"), v_sbyte);
                    }
                    else if (v is byte)
                    {
                        var v_byte = (byte)v;
                        writer.Write("{0}U /* {1} */", v_byte.ToString("x2"), v_byte);
                    }
                    else if (v is short)
                    {
                        var v_short = (short)v;
                        writer.Write("{0} /* {1} */", v_short.ToString("x4"), v_short);
                    }
                    else if (v is ushort)
                    {
                        var v_ushort = (ushort)v;
                        writer.Write("{0}U /* {1} */", v_ushort.ToString("x4"), v_ushort);
                    }
                    else if (v is int)
                    {
                        var v_int = (int)v;
                        writer.Write("{0} /* {1} */", v_int.ToString("x8"), v_int);
                    }
                    else if (v is uint)
                    {
                        var v_uint = (uint)v;
                        writer.Write("{0}U /* {1} */", v_uint.ToString("x8"), v_uint);
                    }
                    else if (v is long)
                    {
                        var v_long = (long)v;
                        writer.Write("{0} /* {1} */", v_long.ToString("x16"), v_long);
                    }
                    else if (v is ulong)
                    {
                        var v_ulong = (ulong)v;
                        writer.Write("{0}U /* {1} */", v_ulong.ToString("x16"), v_ulong);
                    }
                    else if (v is Bit8)
                    {
                        var v_bit8 = (Bit8)v;
                        writer.Write("{0}U", v_bit8.Raw.ToString("x2"), v_bit8);
                    }
                    else if (v is Bit16)
                    {
                        var v_bit16 = (Bit16)v;
                        writer.Write("{0}U", v_bit16.Raw.ToString("x4"), v_bit16);
                    }
                    else if (v is Bit32)
                    {
                        var v_bit32 = (Bit32)v;
                        writer.Write("{0}U", v_bit32.Raw.ToString("x8"), v_bit32);
                    }
                    else if (v is Bit64)
                    {
                        var v_bit64 = (Bit64)v;
                        writer.Write("{0}U", v_bit64.Raw.ToString("x16"), v_bit64);
                    }
                    else if (v is float)
                    {
                        var v_float = (float)v;
                        var bytes = BitConverter.GetBytes(v_float);
                        writer.Write("0f{0}{1}{2}{3} /* {4}f */", bytes[0].ToString("x2"), bytes[1].ToString("x2"), bytes[2].ToString("x2"), bytes[3].ToString("x2"), v_float.ToInvariantString());
                    }
                    else if (v is double)
                    {
                        var v_double = (double)v;
                        var bytes = BitConverter.GetBytes(v_double);
                        writer.Write("0d{0}{1}{2}{3}{4}{5}{6}{7} /* {8}d */", bytes[0].ToString("x2"), bytes[1].ToString("x2"), bytes[2].ToString("x2"), bytes[3].ToString("x2"),
                            bytes[4].ToString("x2"), bytes[5].ToString("x2"), bytes[6].ToString("x2"), bytes[7].ToString("x2"), v_double.ToInvariantString());
                    }
                    else
                    {
                        throw AssertionHelper.Fail();
                    }
                };

                Action<Object> render_scalar_or_vector = sov =>
                {
                    sov.AssertNotNull();

                    var t = (Type)sov.GetType();
                    (t != null && !t.is_arr()).AssertTrue();

                    var vec = sov as IEnumerable;
                    if (vec != null)
                    {
                        writer.Write("{");
                        foreach (var el in vec) render_scalar(el);
                        writer.Write("}");
                    }
                    else
                    {
                        render_scalar(sov);
                    }
                };

                Action<Object> render = o =>
                {
                    o.AssertNotNull();

                    var arr = o as Array;
                    if (arr != null)
                    {
                        // todo. support different flavors of multidimensional arrays
                        (arr.GetType().GetArrayRank() == 1).AssertTrue();
                        arr.GetType().GetElementType().IsArray.AssertFalse();

                        writer.Write("{");
                        foreach (var el in arr) render_scalar_or_vector(el);
                        writer.Write("}");
                    }
                    else
                    {
                        render_scalar_or_vector(o);
                    }
                };

                render(Value);
            }
        }

        protected override void RenderCubin()
        {
            throw new NotImplementedException();
        }
    }
}