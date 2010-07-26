using System.Diagnostics;
using Libptx.Instructions.Annotations;

namespace Libptx.Instructions.Miscellaneous
{
    [Ptxop("trap;")]
    [DebuggerNonUserCode]
    public partial class trap : ptxop
    {
    }
}
