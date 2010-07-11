using System.Diagnostics;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("sub.type                    d, a, b;")]
    [Ptxop("sub{.sat}.s32               d, a, b;")]
    [Ptxop("sub.cc.type                 d, a, b;")]
    [Ptxop("subc{.cc}.type              d, a, b;")]
    [Ptxop("sub{.rnd}{.ftz}{.sat}.f32   d, a, b;")]
    [Ptxop("sub{.rnd}.f64               d, a, b;")]
    [DebuggerNonUserCode]
    internal class sub : ptxop
    {
        [Endian(SoftwareIsa.PTX_13)] public bool c { get; set; }
        [Suffix(SoftwareIsa.PTX_13)] public bool cc { get; set; }
        [Suffix] public frnd rnd { get; set; }
        [Suffix] public bool ftz { get; set; }
        [Suffix] public bool sat { get; set; }
        [Suffix] public type type { get; set; }

        protected override HardwareIsa custom_hwisa
        {
            get
            {
                var f32_rmp = type == f32 && (rnd == rm || rn == rp);
                return f32_rmp ? HardwareIsa.SM_20 : HardwareIsa.SM_10;
            }
        }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (c || cc).AssertImplies(type == s32 || type == u32);
            (c || cc).AssertImplies(sat == false);
            (rnd != null).AssertImplies(type.isfloat());
            (ftz == true).AssertImplies(type == f32);
            (sat == true).AssertImplies(type == s32 || type == f32);
        }
    }
}