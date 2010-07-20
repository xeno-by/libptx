using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libcuda.Versions;
using XenoGears.Assertions;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop("ld{.ss}{.cop}.type          d, [a];")]
    [Ptxop("ld.volatile{.ss}.type       d, [a];")]
    [Ptxop("ldu{.ss}.type               d, [a];")]
    [DebuggerNonUserCode]
    public partial class ld : ptxop
    {
        [Mod] public bool u { get; set; }
        [Affix(SoftwareIsa.PTX_11)] public bool @volatile { get; set; }
        [Affix] public space ss { get; set; }
        [Affix] public cop cop { get; set; }
        [Affix] public Type type { get; set; }

        protected override SoftwareIsa custom_swisa
        {
            get
            {
                var generic = ss == 0;
                var cache = cop != 0;
                return (generic || cache) ? SoftwareIsa.PTX_20 : SoftwareIsa.PTX_11;
            }
        }

        protected override HardwareIsa custom_hwisa
        {
            get
            {
                var generic = ss == 0;
                var cache = cop != 0;
                return (generic || cache) ? HardwareIsa.SM_20 : HardwareIsa.SM_10;
            }
        }

        protected override bool allow_int8 { get { return true; } }
        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override bool allow_vec { get { return true; } }
        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (u == true).AssertImplies(ss == 0 || ss == global);
            (u == true).AssertImplies(@volatile == false);
            (u == true).AssertImplies(cop == 0);
            (@volatile == true).AssertEquiv(cop == 0);
            (cop == 0 || cop == ca || cop == cg || cop == cs || cop == lu || cop == cv).AssertTrue();
            type.isv1().AssertFalse();
        }
    }
}