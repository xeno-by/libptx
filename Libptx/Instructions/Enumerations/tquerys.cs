using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    public class tquerys
    {
        public static tquerys filter_mode { get { throw new NotImplementedException(); } }
        public static tquerys addr_mode_0 { get { throw new NotImplementedException(); } }
        public static tquerys addr_mode_1 { get { throw new NotImplementedException(); } }
        public static tquerys addr_mode_2 { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(tquerys m1, tquerys m2) { throw new NotImplementedException(); }
        public static bool operator !=(tquerys m1, tquerys m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(tquerys tquerys) { throw new NotImplementedException(); }
        public static implicit operator tquerys(String tquerys) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    public static class tquerys_extensions
    {
        public static String name(this tquerys tquerys) { throw new NotImplementedException(); }
    }
}