using System.Diagnostics;
using Libptx.Common.Annotations.Quantas;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;

namespace Libptx.Instructions.ComparisonAndSelection
{
    [Ptxop("slct.dtype.s32          d, a, b, c;")]
    [Ptxop("slct{.ftz}.dtype.f32    d, a, b, c;")]
    [DebuggerNonUserCode]
    internal class slct : ptxop
    {
        [Affix] public bool ftz { get; set; }
        [Affix] public type dtype { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
    }
}