using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    public class mulm
    {
        public static mulm hi { get { throw new NotImplementedException(); } }
        public static mulm lo { get { throw new NotImplementedException(); } }
        public static mulm wide { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(mulm m1, mulm m2) { throw new NotImplementedException(); }
        public static bool operator !=(mulm m1, mulm m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(mulm mulm) { throw new NotImplementedException(); }
        public static implicit operator mulm(String mulm) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    public static class mulm_extensions
    {
        public static String name(this mulm mulm) { throw new NotImplementedException(); }
    }
}