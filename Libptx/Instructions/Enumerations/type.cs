using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    public class type
    {
        public static type u8 { get { throw new NotImplementedException(); } }
        public static type s8 { get { throw new NotImplementedException(); } }
        public static type u16 { get { throw new NotImplementedException(); } }
        public static type s16 { get { throw new NotImplementedException(); } }
        public static type u24 { get { throw new NotImplementedException(); } }
        public static type s24 { get { throw new NotImplementedException(); } }
        public static type u32 { get { throw new NotImplementedException(); } }
        public static type s32 { get { throw new NotImplementedException(); } }
        public static type u64 { get { throw new NotImplementedException(); } }
        public static type s64 { get { throw new NotImplementedException(); } }
        public static type f32 { get { throw new NotImplementedException(); } }
        public static type f64 { get { throw new NotImplementedException(); } }
        public static type b8 { get { throw new NotImplementedException(); } }
        public static type b16 { get { throw new NotImplementedException(); } }
        public static type b32 { get { throw new NotImplementedException(); } }
        public static type b64 { get { throw new NotImplementedException(); } }
        public static type pred { get { throw new NotImplementedException(); } }

        public Type clr { get { throw new NotImplementedException(); } }
        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(type t1, type t2) { throw new NotImplementedException(); }
        public static bool operator !=(type t1, type t2) { return !(t1 == t2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator Type(type type) { throw new NotImplementedException(); }
        public static implicit operator type(Type type) { throw new NotImplementedException(); }

        public static implicit operator String(type type) { throw new NotImplementedException(); }
        public static implicit operator type(String type) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    public static class type_extensions
    {
        public static bool isint(this type type) { throw new NotImplementedException(); }
        public static bool issigned(this type type) { throw new NotImplementedException(); }
        public static bool isunsigned(this type type) { throw new NotImplementedException(); }
        public static bool isfloat(this type type) { throw new NotImplementedException(); }
        public static bool isbit(this type type) { throw new NotImplementedException(); }
        public static bool ispred(this type type) { throw new NotImplementedException(); }

        public static int bits(this type type) { throw new NotImplementedException(); }
        public static bool is8(this type type) { throw new NotImplementedException(); }
        public static bool is16(this type type) { throw new NotImplementedException(); }
        public static bool is24(this type type) { throw new NotImplementedException(); }
        public static bool is32(this type type) { throw new NotImplementedException(); }
        public static bool is64(this type type) { throw new NotImplementedException(); }

        public static Type clr(this type type) { throw new NotImplementedException(); }
        public static String name(this type type) { throw new NotImplementedException(); }
    }
}