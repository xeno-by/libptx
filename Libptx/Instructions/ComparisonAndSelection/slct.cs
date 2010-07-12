using System.Diagnostics;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;

namespace Libptx.Instructions.ComparisonAndSelection
{
    [Ptxop10("slct.dtype.s32          d, a, b, c;")]
    [Ptxop10("slct{.ftz}.dtype.f32    d, a, b, c;")]
    [DebuggerNonUserCode]
    internal class slct : ptxop
    {
        [Infix] public bool ftz { get; set; }
        [Infix] public type dtype { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
    }
}