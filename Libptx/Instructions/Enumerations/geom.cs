using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    public class geom
    {
        public static geom d1 { get { throw new NotImplementedException(); } }
        public static geom d2 { get { throw new NotImplementedException(); } }
        public static geom d3 { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(geom m1, geom m2) { throw new NotImplementedException(); }
        public static bool operator !=(geom m1, geom m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(geom geom) { throw new NotImplementedException(); }
        public static implicit operator geom(String geom) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    public static class geom_extensions
    {
        public static String name(this geom geom) { throw new NotImplementedException(); }
    }
}