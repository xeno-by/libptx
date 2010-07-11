using System.Diagnostics;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Instructions.DataMovementAndConversion
{
    [Ptxop("st{.ss}{.cop}.type          d, [a];")]
    [Ptxop("st{.ss}{.cop}.vec.type      d, [a];")]
    [Ptxop("st.volatile{.ss}.type       d, [a];")]
    [Ptxop("st.volatile{.ss}.vec.type   d, [a];")]
    [DebuggerNonUserCode]
    internal class st : ptxop
    {
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
            (@volatile == true).AssertEquiv(cop == null);
            (ss == null || ss == wb || ss == cg || ss == cs || ss == wt).AssertTrue();
        }
    }
}