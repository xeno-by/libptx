using System.Diagnostics;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;

namespace Libptx.Instructions.DataMovementAndConversion
{
    [Ptxop("mov.type d, a;")]
    [Ptxop("mov.type d, sreg;")]
    [Ptxop("mov.type d, avar;")]
    [Ptxop("mov.type d, label;")]
    [DebuggerNonUserCode]
    internal class mov : ptxop
    {
        [Suffix] public type type { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override bool allow_pred { get { return true; } }
    }
}