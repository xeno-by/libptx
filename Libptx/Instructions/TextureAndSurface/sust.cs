using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Edsl.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;
using Type = Libptx.Common.Types.Type;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop("sust.b.geom{.cop}.dtype.clampm    d, [a, b];", SoftwareIsa.PTX_15)]
    [Ptxop("sust.p.geom{.cop}.dtype.clampm    d, [a, b];", SoftwareIsa.PTX_20)]
    [DebuggerNonUserCode]
    public class sust : ptxop
    {
        [Affix] public bool b { get; set; }
        [Affix] public bool p { get; set; }
        [Affix] public geom geom { get; set; }
        [Affix] public cop cop { get; set; }
        [Affix] public Type ctype { get; set; }
        [Affix] public clampm clampm { get; set; }

        protected override SoftwareIsa custom_swisa
        {
            get
            {
                var cache = cop != 0;
                var sust_p = p == true;
                var sust_3d = geom == d3;
                var non_trap = clampm != clamp_trap;
                return cache || sust_p || sust_3d || non_trap ? SoftwareIsa.PTX_20 : SoftwareIsa.PTX_15;
            }
        }

        protected override HardwareIsa custom_hwisa
        {
            get
            {
                var cache = cop != 0;
                var sust_p = p == true;
                var sust_3d = geom == d3;
                var non_trap = clampm != clamp_trap;
                return cache || sust_p || sust_3d || non_trap ? HardwareIsa.SM_20 : HardwareIsa.SM_10;
            }
        }

        protected override bool allow_bit8 { get { return true; } }
        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (b || p).AssertTrue();
            (geom != 0).AssertTrue();
            (cop == 0 || cop == wb || cop == cg || cop == cs || cop == wt).AssertTrue();

            (b == true).AssertImplies(ctype.isbit());
            (b == true).AssertImplies(ctype.isscalar() || ctype.isv2() || ctype.isv4());
            (p == true).AssertImplies(ctype.is32());
            (p == true).AssertImplies(ctype.isv4());
        }
    }
}