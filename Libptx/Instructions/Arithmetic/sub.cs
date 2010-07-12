using System.Diagnostics;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop10("sub.type                    d, a, b;")]
    [Ptxop10("sub{.sat}.s32               d, a, b;")]
    [Ptxop10("sub.cc.type                 d, a, b;")]
    [Ptxop10("subc{.cc}.type              d, a, b;")]
    [Ptxop10("sub{.rnd}{.ftz}{.sat}.f32   d, a, b;")]
    [Ptxop10("sub{.rnd}.f64               d, a, b;")]
    [DebuggerNonUserCode]
    internal class sub : ptxop
    {
        [Mod(SoftwareIsa.PTX_13)] public bool c { get; set; }
        [Infix(SoftwareIsa.PTX_13)] public bool cc { get; set; }
        [Infix] public frnd rnd { get; set; }
        [Infix] public bool ftz { get; set; }
        [Infix] public bool sat { get; set; }
        [Infix] public type type { get; set; }

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