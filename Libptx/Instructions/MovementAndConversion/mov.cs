using System.Diagnostics;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop10("mov.type d, a;")]
    [Ptxop10("mov.type d, sreg;")]
    [Ptxop10("mov.type d, avar;")]
    [Ptxop10("mov.type d, label;")]
    [DebuggerNonUserCode]
    internal class mov : ptxop
    {
        [Infix] public type type { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override bool allow_pred { get { return true; } }
    }
}