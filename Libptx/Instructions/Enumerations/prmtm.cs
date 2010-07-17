using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    public class prmtm
    {
        public static prmtm f4e { get { throw new NotImplementedException(); } }
        public static prmtm b4e { get { throw new NotImplementedException(); } }
        public static prmtm rc8 { get { throw new NotImplementedException(); } }
        public static prmtm ec1 { get { throw new NotImplementedException(); } }
        public static prmtm ecr { get { throw new NotImplementedException(); } }
        public static prmtm rc16 { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(prmtm m1, prmtm m2) { throw new NotImplementedException(); }
        public static bool operator !=(prmtm m1, prmtm m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(prmtm prmtm) { throw new NotImplementedException(); }
        public static implicit operator prmtm(String prmtm) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    public static class prmtm_extensions
    {
        public static String name(this prmtm prmtm) { throw new NotImplementedException(); }
    }
}