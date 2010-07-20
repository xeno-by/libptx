using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;

namespace Libptx.Instructions.ComparisonAndSelection
{
    [Ptxop("selp.type d, a, b, c;")]
    [DebuggerNonUserCode]
    public partial class selp : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
    }
}