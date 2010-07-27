using System.Diagnostics;
using Libptx.Common.Enumerations;
using Libptx.Common.Annotations.Quanta;
using Libptx.Expressions.Sregs.Annotations;
using XenoGears.Assertions;

namespace Libptx.Expressions.Sregs
{
    [Special20("%lanemask_{op}", typeof(uint))]
    [DebuggerNonUserCode]
    public partial class lanemask : Sreg
    {
        [Affix("op")] public cmp Mask { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            (Mask == eq || Mask == gt || Mask == ge || Mask == lt || Mask == le).AssertTrue();
        }
    }
}