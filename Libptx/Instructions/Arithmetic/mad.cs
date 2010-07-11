using System;
using System.Diagnostics;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("mad{.mode}.type         d, a, b, c;")]
    [Ptxop("mad.hi.sat.s32          d, a, b, c;")]
    [Ptxop("mad24{.hi,.lo}.type     d, a, b, c;")]
    [Ptxop("mad24.hi.sat.s32        d, a, b, c;")]
    [Ptxop("mad{.ftz}{.sat}.f32     d, a, b, c;")]
    [Ptxop("mad.rnd{.ftz}{.sat}.f32 d, a, b, c;")]
    [Ptxop("mad.rnd.f64             d, a, b, c;")]
    [DebuggerNonUserCode]
    internal class mad : ptxop
    {
        [Suffix] public mulm mode { get; set; }
        [Suffix] public frnd rnd { get; set; }
        [Suffix] public bool ftz { get; set; }
        [Suffix] public bool sat { get; set; }
        [Suffix] public type type { get; set; }

        protected override HardwareIsa custom_hwisa
        {
            get
            {
                var f32_rnd = type == f32 && rnd != null;
                return f32_rnd ? HardwareIsa.SM_20 : HardwareIsa.SM_10;
            }
        }

        protected override bool allow_int24 { get { return true; } }
        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (mode != null).AssertImplies(type.isint());
            (mode == wide).AssertImplies(type.is16() || type.is32());
            (rnd != null).AssertImplies(type.isfloat());
            (type == f64).AssertImplies(rnd != null);
            (ftz == true).AssertImplies(type == f32);
            (sat == true).AssertImplies(type == s24 || type == s32 || type == f32);
            (sat == true && type.isint()).AssertImplies(mode == mulm_hi);

            (target_swisa >= SoftwareIsa.PTX_14 && type == f64).AssertImplies(rnd != null);
            (target_hwisa >= HardwareIsa.SM_20 && type == f32).AssertImplies(rnd != null);
        }

        protected override String to_string()
        {
            var to_s = base.to_string();
            if (type.is24()) to_s = to_s.Replace("24", "32");
            if (type.is24()) to_s = to_s.Replace("mul", "mul24");
            return to_s;
        }
    }
}