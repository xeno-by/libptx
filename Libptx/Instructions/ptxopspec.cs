using System;
using System.Diagnostics;

namespace Libptx.Instructions
{
    [DebuggerNonUserCode]
    internal class ptxopspec
    {
        public static implicit operator ptxopspec(String spec)
        {
            throw new NotImplementedException();
        }

        public static implicit operator String(ptxopspec spec) { throw new NotImplementedException(); }
        public override String ToString() { return (String)this; }
    }
}