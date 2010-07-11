using System.Diagnostics;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop("ld{.ss}{.cop}.type          d, [a];")]
    [Ptxop("ld{.ss}{.cop}.vec.type      d, [a];")]
    [Ptxop("ld.volatile{.ss}.type       d, [a];")]
    [Ptxop("ld.volatile{.ss}.vec.type   d, [a];")]
    [Ptxop("ldu{.ss}.type               d, [a];")]
    [Ptxop("ldu{.ss}.vec.type           d, [a];")]
    [DebuggerNonUserCode]
    internal class ld : ptxop
    {
        [Endian] public bool u { get; set; }
        [Suffix(SoftwareIsa.PTX_11)] public bool @volatile { get; set; }
        [Suffix] public ss ss { get; set; }
        [Suffix] public cop cop { get; set; }
        [Suffix] public vec vec { get; set; }
        [Suffix] public type type { get; set; }

        protected override SoftwareIsa custom_swisa
        {
            get
            {
                var generic = ss == null;
                var cache = cop != null;
                return (generic || cache) ? SoftwareIsa.PTX_20 : SoftwareIsa.PTX_11;
            }
        }

        protected override HardwareIsa custom_hwisa
        {
            get
            {
                var generic = ss == null;
                var cache = cop != null;
                return (generic || cache) ? HardwareIsa.SM_20 : HardwareIsa.SM_10;
            }
        }

        protected override bool allow_int8 { get { return true; } }
        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (u == true).AssertImplies(ss == null || ss == global);
            (u == true).AssertImplies(@volatile == false);
            (u == true).AssertImplies(cop == null);
            (@volatile == true).AssertEquiv(cop == null);
            (ss == null || ss == ca || ss == cg || ss == cs || ss == lu || ss == cv).AssertTrue();
        }
    }
}