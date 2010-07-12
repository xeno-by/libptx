using System;
using System.Diagnostics;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop10("suld.b.geom{.cop}.vec.dtype.clampm   d, [a, b];", SoftwareIsa.PTX_15)]
    [Ptxop10("suld.p.geom{.cop}.v4.dtype.clampm    d, [a, b];", SoftwareIsa.PTX_20)]
    [DebuggerNonUserCode]
    internal class suld : ptxop
    {
        [Infix] public bool b { get; set; }
        [Infix] public bool p { get; set; }
        [Infix] public geom geom { get; set; }
        [Infix] public cop cop { get; set; }
        [Infix] public vec vec { get; set; }
        [Infix] public type dtype { get; set; }
        [Infix] public clampm clampm { get; set; }

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
            (cop == null || cop == ca || cop == cg || cop == cs || cop == cv).AssertTrue();
            (p == true).AssertImplies(vec == v4);
            (b == true).AssertImplies(dtype.isbit());
            (p == true).AssertImplies(dtype.is32());
        }
    }
}