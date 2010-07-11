using System.Diagnostics;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop("sust.b.geom{.cop}.vec.dtype.clampm   d, [a, b];", SoftwareIsa.PTX_15)]
    [Ptxop("sust.p.geom{.cop}.v4.dtype.clampm    d, [a, b];", SoftwareIsa.PTX_20)]
    [DebuggerNonUserCode]
    internal class sust : ptxop
    {
        [Suffix] public bool b { get; set; }
        [Suffix] public bool p { get; set; }
        [Suffix] public geom geom { get; set; }
        [Suffix] public cop cop { get; set; }
        [Suffix] public vec vec { get; set; }
        [Suffix] public type ctype { get; set; }
        [Suffix] public clampm clampm { get; set; }

        protected override SoftwareIsa custom_swisa
        {
            get
            {
                var cache = cop != null;
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
                var cache = cop != null;
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
        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (b || p).AssertTrue();
            (geom != null).AssertTrue();
            (cop == null || cop == wb || cop == cg || cop == cs || cop == wt).AssertTrue();
            (vec != null).AssertTrue();
            (b == true).AssertImplies(ctype.isbit());
            (p == true).AssertImplies(ctype.is32());
        }
    }
}