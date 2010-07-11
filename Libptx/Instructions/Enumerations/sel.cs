using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    internal class sel
    {
        public static sel b0 { get { throw new NotImplementedException(); } }
        public static sel b1 { get { throw new NotImplementedException(); } }
        public static sel b2 { get { throw new NotImplementedException(); } }
        public static sel b3 { get { throw new NotImplementedException(); } }
        public static sel h0 { get { throw new NotImplementedException(); } }
        public static sel h1 { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(sel m1, sel m2) { throw new NotImplementedException(); }
        public static bool operator !=(sel m1, sel m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(sel sel) { throw new NotImplementedException(); }
        public static implicit operator sel(String sel) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    internal static class sel_extensions
    {
        public static String name(this sel sel) { throw new NotImplementedException(); }
    }
}