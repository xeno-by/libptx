using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    public class cachelevel
    {
        public static cachelevel L1 { get { throw new NotImplementedException(); } }
        public static cachelevel L2 { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(cachelevel m1, cachelevel m2) { throw new NotImplementedException(); }
        public static bool operator !=(cachelevel m1, cachelevel m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(cachelevel cachelevel) { throw new NotImplementedException(); }
        public static implicit operator cachelevel(String cachelevel) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    public static class cachelevel_extensions
    {
        public static String name(this cachelevel cachelevel) { throw new NotImplementedException(); }
    }
}