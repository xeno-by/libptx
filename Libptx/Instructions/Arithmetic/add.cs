using System.Diagnostics;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("add.type                    d, a, b;")]
    [Ptxop("add{.sat}.s32               d, a, b;")]
    [Ptxop("add.cc.type                 d, a, b;")]
    [Ptxop("addc{.cc}.type              d, a, b;")]
    [Ptxop("add{.rnd}{.ftz}{.sat}.f32   d, a, b;")]
    [Ptxop("add{.rnd}.f64               d, a, b;")]
    [DebuggerNonUserCode]
    internal class add : ptxop
    {
        [Endian(SoftwareIsa.PTX_12)] public bool c { get; set; }
        [Suffix(SoftwareIsa.PTX_12)] public bool cc { get; set; }
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
