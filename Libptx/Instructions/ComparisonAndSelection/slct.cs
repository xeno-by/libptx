using System.Diagnostics;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;

namespace Libptx.Instructions.ComparisonAndSelection
{
    [Ptxop("slct.dtype.s32          d, a, b, c;")]
    [Ptxop("slct{.ftz}.dtype.f32    d, a, b, c;")]
    [DebuggerNonUserCode]
    internal class slct : ptxop
    {
        [Suffix] public bool ftz { get; set; }
        [Suffix] public type dtype { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
    }
}