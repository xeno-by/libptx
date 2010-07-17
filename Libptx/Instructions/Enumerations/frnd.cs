using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    public class frnd
    {
        public static frnd rn { get { throw new NotImplementedException(); } }
        public static frnd rz { get { throw new NotImplementedException(); } }
        public static frnd rm { get { throw new NotImplementedException(); } }
        public static frnd rp { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(frnd r1, frnd r2) { throw new NotImplementedException(); }
        public static bool operator !=(frnd r1, frnd r2) { return !(r1 == r2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(frnd frnd) { throw new NotImplementedException(); }
        public static implicit operator frnd(String frnd) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    public static class frnd_extensions
    {
        public static String name(this frnd frnd) { throw new NotImplementedException(); }
    }
}