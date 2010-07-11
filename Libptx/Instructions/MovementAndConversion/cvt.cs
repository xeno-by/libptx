using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop("cvt{.irnd}{.ftz}{.sat}.dtype.atype d, a;")]
    [Ptxop("cvt{.frnd}{.ftz}{.sat}.dtype.atype d, a;")]
    [DebuggerNonUserCode]
    internal class cvt : ptxop
    {
        [Suffix] public irnd irnd { get; set; }
        [Suffix] public frnd frnd { get; set; }
        [Suffix] public bool ftz { get; set; }
        [Suffix] public bool sat { get; set; }
        [Suffix] public type dtype { get; set; }
        [Suffix] public type atype { get; set; }

        protected override bool allow_int8 { get { return true; } }
        protected override bool allow_float16 { get { return true; } }
        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            var i2i = atype.isint() && dtype.isint();
            var i2f = atype.isint() && dtype.isfloat();
            var f2i = atype.isfloat() && dtype.isint();
            var f2f = atype.isfloat() && dtype.isfloat();
            (i2i || i2f || f2i || f2f).AssertTrue();

            (irnd != null).AssertImplies(f2i || f2f);
            (frnd != null).AssertImplies(i2f || f2f);
            (irnd != null && frnd != null).AssertFalse();
        }
    }
}
