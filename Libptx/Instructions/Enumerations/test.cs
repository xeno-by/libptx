using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    public class test
    {
        public static test finite { get { throw new NotImplementedException(); } }
        public static test infinite { get { throw new NotImplementedException(); } }
        public static test number { get { throw new NotImplementedException(); } }
        public static test notanumber { get { throw new NotImplementedException(); } }
        public static test normal { get { throw new NotImplementedException(); } }
        public static test subnormal { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(test m1, test m2) { throw new NotImplementedException(); }
        public static bool operator !=(test m1, test m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(test test) { throw new NotImplementedException(); }
        public static implicit operator test(String testpop) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    public static class testpop_extensions
    {
        public static String name(this test test) { throw new NotImplementedException(); }
    }
}