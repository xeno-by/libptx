using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    internal class cmpop
    {
        public static cmpop eq { get { throw new NotImplementedException(); } }
        public static cmpop ne { get { throw new NotImplementedException(); } }
        public static cmpop lt { get { throw new NotImplementedException(); } }
        public static cmpop le { get { throw new NotImplementedException(); } }
        public static cmpop gt { get { throw new NotImplementedException(); } }
        public static cmpop ge { get { throw new NotImplementedException(); } }
        public static cmpop lo { get { throw new NotImplementedException(); } }
        public static cmpop ls { get { throw new NotImplementedException(); } }
        public static cmpop hi { get { throw new NotImplementedException(); } }
        public static cmpop hs { get { throw new NotImplementedException(); } }
        public static cmpop equ { get { throw new NotImplementedException(); } }
        public static cmpop neu { get { throw new NotImplementedException(); } }
        public static cmpop ltu { get { throw new NotImplementedException(); } }
        public static cmpop leu { get { throw new NotImplementedException(); } }
        public static cmpop gtu { get { throw new NotImplementedException(); } }
        public static cmpop geu { get { throw new NotImplementedException(); } }
        public static cmpop num { get { throw new NotImplementedException(); } }
        public static cmpop nan { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(cmpop m1, cmpop m2) { throw new NotImplementedException(); }
        public static bool operator !=(cmpop m1, cmpop m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(cmpop cmpop) { throw new NotImplementedException(); }
        public static implicit operator cmpop(String cmpop) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    internal static class cmpop_extensions
    {
        public static String name(this cmpop cmpop) { throw new NotImplementedException(); }
    }
}