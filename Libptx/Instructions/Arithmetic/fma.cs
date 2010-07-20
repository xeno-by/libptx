using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("fma.rnd{.ftz}{.sat}.f32 d, a, b, c;")]
    [Ptxop("fma.rnd.f64             d, a, b, c;")]
    [DebuggerNonUserCode]
    public partial class fma : ptxop
    {
        [Affix] public frnd rnd { get; set; }
        [Affix] public bool ftz { get; set; }
        [Affix] public bool sat { get; set; }
        [Affix] public Type type { get; set; }

        protected override SoftwareIsa custom_swisa
        {
            get { return type == f32 ? SoftwareIsa.PTX_20 : SoftwareIsa.PTX_10; }
        }

        protected override HardwareIsa custom_hwisa
        {
            get { return type == f32 ? HardwareIsa.SM_20 : HardwareIsa.SM_10; }
        }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (rnd != 0).AssertTrue();
            (ftz == true).AssertImplies(type == f32);
            (sat == true).AssertImplies(type == f32);
            type.isfloat().AssertTrue();
        }
    }
}