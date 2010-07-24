using System.Diagnostics;
using Libptx.Instructions.Annotations;

namespace Libptx.Instructions.ControlFlow
{
    [Ptxop("exit;")]
    [DebuggerNonUserCode]
    public partial class exit : ptxop
    {
    }
}