using System;
using System.Diagnostics;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop10("mul{.mode}.type             d, a, b;")]
    [Ptxop10("mul24{.hi,.lo}.type         d, a, b;")]
    [Ptxop10("mul{.rnd}{.ftz}{.sat}.f32   d, a, b;")]
    [Ptxop10("mul{.rnd}.f64               d, a, b;")]
    [DebuggerNonUserCode]
    internal class mul : ptxop
    {
        [Infix] public mulm mode { get; set; }
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

        protected override bool allow_int24 { get { return true; } }
        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (mode != null).AssertImplies(type.isint());
            (mode == wide).AssertImplies(type.is16() || type.is32());
            (rnd != null).AssertImplies(type.isfloat());
            (ftz == true).AssertImplies(type == f32);
            (sat == true).AssertImplies(type == f32);
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