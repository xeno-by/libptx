using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Expressions.Sregs.Annotations;
using XenoGears.Assertions;

namespace Libptx.Expressions.Sregs
{
    [Special("%pm{index}", typeof(uint), SoftwareIsa.PTX_13)]
    [DebuggerNonUserCode]
    public partial class pm : Sreg
    {
        [Affix("index")] public int Index { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            (0 <= Index && Index <= 3).AssertTrue();
        }
    }
}