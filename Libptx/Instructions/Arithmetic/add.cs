using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
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
    public class add : ptxop
    {
        [Mod(SoftwareIsa.PTX_12)] public bool c { get; set; }
        [Affix(SoftwareIsa.PTX_12)] public bool cc { get; set; }
        [Affix] public frnd rnd { get; set; }
        [Affix] public bool ftz { get; set; }
        [Affix] public bool sat { get; set; }
        [Affix] public type type { get; set; }

        protected override HardwareIsa custom_hwisa
        {
            get
            {
                var f32_rmp = type == f32 && (rnd == rm || rn == rp);
                return f32_rmp ? HardwareIsa.SM_20 : HardwareIsa.SM_10;
            }
        }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (c || cc).AssertImplies(type == s32 || type == u32);
            (c || cc).AssertImplies(sat == false);
            (rnd != null).AssertImplies(type.isfloat());
            (ftz == true).AssertImplies(type == f32);
            (sat == true).AssertImplies(type == s32 || type == f32);
        }
    }
}
