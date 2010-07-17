using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    public class vec
    {
        public static vec v2 { get { throw new NotImplementedException(); } }
        public static vec v4 { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(vec r1, vec r2) { throw new NotImplementedException(); }
        public static bool operator !=(vec r1, vec r2) { return !(r1 == r2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(vec vec) { throw new NotImplementedException(); }
        public static implicit operator vec(String vec) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    public static class vec_extensions
    {
        public static String name(this vec vec) { throw new NotImplementedException(); }
    }
}