using System.Diagnostics;
using Libptx.Expressions.Sregs.Annotations;

namespace Libptx.Expressions.Sregs
{
    [Sreg20("%nsmid", typeof(uint))]
    [DebuggerNonUserCode]
    public partial class nsmid : Sreg
    {
    }
}