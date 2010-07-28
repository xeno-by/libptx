using System.Diagnostics;
using Libptx.Expressions.Sregs.Annotations;

namespace Libptx.Expressions.Sregs
{
    [Sreg20("%nwarpid", typeof(uint))]
    [DebuggerNonUserCode]
    public partial class nwarpid : Sreg
    {
    }
}