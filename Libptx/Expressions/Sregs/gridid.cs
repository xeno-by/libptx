using System.Diagnostics;
using Libptx.Expressions.Sregs.Annotations;

namespace Libptx.Expressions.Sregs
{
    [Special("%gridid", typeof(uint))]
    [DebuggerNonUserCode]
    public partial class gridid : Sreg
    {
    }
}