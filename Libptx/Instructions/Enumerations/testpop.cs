using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    internal class testpop
    {
        public static testpop finite { get { throw new NotImplementedException(); } }
        public static testpop infinite { get { throw new NotImplementedException(); } }
        public static testpop number { get { throw new NotImplementedException(); } }
        public static testpop notanumber { get { throw new NotImplementedException(); } }
        public static testpop normal { get { throw new NotImplementedException(); } }
        public static testpop subnormal { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(testpop m1, testpop m2) { throw new NotImplementedException(); }
        public static bool operator !=(testpop m1, testpop m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(testpop testpop) { throw new NotImplementedException(); }
        public static implicit operator testpop(String testpop) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    internal static class testpop_extensions
    {
        public static String name(this testpop testpop) { throw new NotImplementedException(); }
    }
}