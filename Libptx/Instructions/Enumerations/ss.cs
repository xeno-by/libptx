using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    public class ss
    {
        public static ss @const { get { throw new NotImplementedException(); } }
        public static ss global { get { throw new NotImplementedException(); } }
        public static ss local { get { throw new NotImplementedException(); } }
        public static ss param { get { throw new NotImplementedException(); } }
        public static ss shared { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(ss r1, ss r2) { throw new NotImplementedException(); }
        public static bool operator !=(ss r1, ss r2) { return !(r1 == r2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(ss ss) { throw new NotImplementedException(); }
        public static implicit operator ss(String ss) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    public static class ss_extensions
    {
        public static String name(this ss ss) { throw new NotImplementedException(); }
    }
}