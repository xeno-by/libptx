using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    public class barlevel
    {
        public static barlevel cta { get { throw new NotImplementedException(); } }
        public static barlevel gl { get { throw new NotImplementedException(); } }
        public static barlevel sys { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(barlevel m1, barlevel m2) { throw new NotImplementedException(); }
        public static bool operator !=(barlevel m1, barlevel m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(barlevel barlevel) { throw new NotImplementedException(); }
        public static implicit operator barlevel(String barlevel) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    public static class barlevel_extensions
    {
        public static String name(this barlevel barlevel) { throw new NotImplementedException(); }
    }
}