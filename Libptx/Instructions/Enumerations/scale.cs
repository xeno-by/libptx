using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    internal class scale
    {
        public static scale shr7 { get { throw new NotImplementedException(); } }
        public static scale shr15 { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(scale m1, scale m2) { throw new NotImplementedException(); }
        public static bool operator !=(scale m1, scale m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(scale scale) { throw new NotImplementedException(); }
        public static implicit operator scale(String scale) { throw new NotImplementedException(); }
        public static implicit operator int(scale scale) { throw new NotImplementedException(); }
        public static implicit operator scale(int scale) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    internal static class scale_extensions
    {
        public static String name(this scale scale) { throw new NotImplementedException(); }
    }
}