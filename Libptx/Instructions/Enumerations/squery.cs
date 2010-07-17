using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    public class squery
    {
        public static squery width { get { throw new NotImplementedException(); } }
        public static squery height { get { throw new NotImplementedException(); } }
        public static squery depth { get { throw new NotImplementedException(); } }
        public static squery channel_datatype { get { throw new NotImplementedException(); } }
        public static squery channel_order { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(squery m1, squery m2) { throw new NotImplementedException(); }
        public static bool operator !=(squery m1, squery m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(squery squery) { throw new NotImplementedException(); }
        public static implicit operator squery(String squery) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    public static class squery_extensions
    {
        public static String name(this squery squery) { throw new NotImplementedException(); }
    }
}