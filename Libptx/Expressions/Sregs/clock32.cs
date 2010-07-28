using System.Diagnostics;
using Libptx.Expressions.Sregs.Annotations;

namespace Libptx.Expressions.Sregs
{
    [Sreg("%clock", typeof(uint))]
    [DebuggerNonUserCode]
    public partial class clock32 : Sreg
    {
    }
}