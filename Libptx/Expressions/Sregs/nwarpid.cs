using System.Diagnostics;
using Libptx.Expressions.Sregs.Annotations;

namespace Libptx.Expressions.Sregs
{
    [Special20("%nwarpid", typeof(uint))]
    [DebuggerNonUserCode]
    public partial class nwarpid : Sreg
    {
    }
}