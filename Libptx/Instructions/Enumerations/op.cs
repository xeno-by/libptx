using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    internal class op
    {
        public static op add { get { throw new NotImplementedException(); } }
        public static op min { get { throw new NotImplementedException(); } }
        public static op max { get { throw new NotImplementedException(); } }
        public static op and { get { throw new NotImplementedException(); } }
        public static op or { get { throw new NotImplementedException(); } }
        public static op xor { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(op m1, op m2) { throw new NotImplementedException(); }
        public static bool operator !=(op m1, op m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(op op) { throw new NotImplementedException(); }
        public static implicit operator op(String op) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    internal static class op_extensions
    {
        public static String name(this op op) { throw new NotImplementedException(); }
    }
}