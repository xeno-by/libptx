using System.Diagnostics;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;

namespace Libptx.Instructions.ComparisonAndSelection
{
    [Ptxop("selp.type d, a, b, c;")]
    [DebuggerNonUserCode]
    internal class selp : ptxop
    {
        [Suffix] public type type { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
    }
}