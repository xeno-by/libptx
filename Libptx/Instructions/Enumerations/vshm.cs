using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    internal class vshm
    {
        public static vshm clamp { get { throw new NotImplementedException(); } }
        public static vshm wrap { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(vshm m1, vshm m2) { throw new NotImplementedException(); }
        public static bool operator !=(vshm m1, vshm m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(vshm vshm) { throw new NotImplementedException(); }
        public static implicit operator vshm(String vshm) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    internal static class vshm_extensions
    {
        public static String name(this vshm vshm) { throw new NotImplementedException(); }
    }
}