using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    internal class cop
    {
        public static cop ca { get { throw new NotImplementedException(); } }
        public static cop cg { get { throw new NotImplementedException(); } }
        public static cop cs { get { throw new NotImplementedException(); } }
        public static cop lu { get { throw new NotImplementedException(); } }
        public static cop cv { get { throw new NotImplementedException(); } }
        public static cop wb { get { throw new NotImplementedException(); } }
        public static cop wt { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(cop r1, cop r2) { throw new NotImplementedException(); }
        public static bool operator !=(cop r1, cop r2) { return !(r1 == r2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(cop cop) { throw new NotImplementedException(); }
        public static implicit operator cop(String cop) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    internal static class rcop_extensions
    {
        public static String name(this cop cop) { throw new NotImplementedException(); }
    }
}