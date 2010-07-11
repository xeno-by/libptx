using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    internal class redm
    {
        public static redm all { get { throw new NotImplementedException(); } }
        public static redm any { get { throw new NotImplementedException(); } }
        public static redm uni { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(redm m1, redm m2) { throw new NotImplementedException(); }
        public static bool operator !=(redm m1, redm m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(redm redm) { throw new NotImplementedException(); }
        public static implicit operator redm(String redm) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    internal static class redm_extensions
    {
        public static String name(this redm redm) { throw new NotImplementedException(); }
    }
}