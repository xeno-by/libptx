using System.Diagnostics;
using Libptx.Expressions.Sregs.Annotations;

namespace Libptx.Expressions.Sregs
{
    [Sreg20("%clock64", typeof(ulong))]
    [DebuggerNonUserCode]
    public partial class clock64 : Sreg
    {
    }
}