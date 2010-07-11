using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    internal class irnd
    {
        public static irnd rni { get { throw new NotImplementedException(); } }
        public static irnd rzi { get { throw new NotImplementedException(); } }
        public static irnd rmi { get { throw new NotImplementedException(); } }
        public static irnd rpi { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(irnd r1, irnd r2) { throw new NotImplementedException(); }
        public static bool operator !=(irnd r1, irnd r2) { return !(r1 == r2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(irnd irnd) { throw new NotImplementedException(); }
        public static implicit operator irnd(String irnd) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    internal static class irnd_extensions
    {
        public static String name(this irnd irnd) { throw new NotImplementedException(); }
    }
}