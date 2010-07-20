using System;
using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Edsl.Types;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;
using Type = Libptx.Common.Types.Type;

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
    public class mad : ptxop
    {
        [Mod("24")] public bool is24 { get; set; }
        [Affix] public mulm mode { get; set; }
        [Affix] public frnd rnd { get; set; }
        [Affix] public bool ftz { get; set; }
        [Affix] public bool sat { get; set; }
        [Affix] public Type type { get; set; }

        protected override HardwareIsa custom_hwisa
        {
            get
            {
                var f32_rnd = type == f32 && rnd != null;
                return f32_rnd ? HardwareIsa.SM_20 : HardwareIsa.SM_10;
            }
        }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (is24 == true).AssertImplies(type == s32 || type == u32);
            (mode != null).AssertImplies(type.isint());
            (mode == wide).AssertImplies(type.is16() || type.is32());
            (rnd != null).AssertImplies(type.isfloat());
            (type == f64).AssertImplies(rnd != null);
            (ftz == true).AssertImplies(type == f32);
            (sat == true).AssertImplies(type == s32 || type == f32);
            (sat == true && type.isint()).AssertImplies(mode == mulm_hi);

            (target_swisa >= SoftwareIsa.PTX_14 && type == f64).AssertImplies(rnd != null);
            (target_hwisa >= HardwareIsa.SM_20 && type == f32).AssertImplies(rnd != null);
        }
    }
}