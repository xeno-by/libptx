using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    public class tquery
    {
        public static tquery width { get { throw new NotImplementedException(); } }
        public static tquery height { get { throw new NotImplementedException(); } }
        public static tquery depth { get { throw new NotImplementedException(); } }
        public static tquery channel_datatype { get { throw new NotImplementedException(); } }
        public static tquery channel_order { get { throw new NotImplementedException(); } }
        public static tquery normalized_coords { get { throw new NotImplementedException(); } }
        public static tquery filter_mode { get { throw new NotImplementedException(); } }
        public static tquery addr_mode_0 { get { throw new NotImplementedException(); } }
        public static tquery addr_mode_1 { get { throw new NotImplementedException(); } }
        public static tquery addr_mode_2 { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(tquery m1, tquery m2) { throw new NotImplementedException(); }
        public static bool operator !=(tquery m1, tquery m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(tquery tquery) { throw new NotImplementedException(); }
        public static implicit operator tquery(String tquery) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    public static class tquery_extensions
    {
        public static String name(this tquery tquery) { throw new NotImplementedException(); }
    }
}