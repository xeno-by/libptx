using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    public class size
    {
        public static size sz32 { get { throw new NotImplementedException(); } }
        public static size sz64 { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(size m1, size m2) { throw new NotImplementedException(); }
        public static bool operator !=(size m1, size m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(size size) { throw new NotImplementedException(); }
        public static implicit operator size(String size) { throw new NotImplementedException(); }
        public static implicit operator int(size size) { throw new NotImplementedException(); }
        public static implicit operator size(int size) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    public static class size_extensions
    {
        public static String name(this size size) { throw new NotImplementedException(); }
    }
}