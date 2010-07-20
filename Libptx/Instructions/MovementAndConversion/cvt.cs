using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop("cvt{.irnd}{.ftz}{.sat}.dtype.atype d, a;")]
    [Ptxop("cvt{.frnd}{.ftz}{.sat}.dtype.atype d, a;")]
    [DebuggerNonUserCode]
    public partial class cvt : ptxop
    {
        [Affix] public irnd irnd { get; set; }
        [Affix] public frnd frnd { get; set; }
        [Affix] public bool ftz { get; set; }
        [Affix] public bool sat { get; set; }
        [Affix] public Type dtype { get; set; }
        [Affix] public Type atype { get; set; }

        protected override bool allow_int8 { get { return true; } }
        protected override bool allow_float16 { get { return true; } }
        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            var i2i = atype.isint() && dtype.isint();
            var i2f = atype.isint() && dtype.isfloat();
            var f2i = atype.isfloat() && dtype.isint();
            var f2f = atype.isfloat() && dtype.isfloat();
            (i2i || i2f || f2i || f2f).AssertTrue();

            (irnd != 0).AssertImplies(f2i || f2f);
            (frnd != 0).AssertImplies(i2f || f2f);
            (irnd != 0 && frnd != 0).AssertFalse();
        }
    }
}
