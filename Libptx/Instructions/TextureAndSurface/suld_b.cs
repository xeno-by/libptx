using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;
using Type = Libptx.Common.Types.Type;
using Libptx.Expressions;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop("suld.b.geom{.cop}.dtype.clampm    d, [a, b];", SoftwareIsa.PTX_15)]
    [DebuggerNonUserCode]
    public partial class suld_b : ptxop
    {
        [Affix] public geom geom { get; set; }
        [Affix] public cop cop { get; set; }
        [Affix] public Type dtype { get; set; }
        [Affix] public clampm clampm { get; set; }

        protected override SoftwareIsa custom_swisa
        {
            get
            {
                var cache = cop != 0;
                var suld_3d = geom == d3;
                var non_trap = clampm != clamp_trap;
                return cache || suld_3d || non_trap ? SoftwareIsa.PTX_20 : SoftwareIsa.PTX_15;
            }
        }

        protected override HardwareIsa custom_hwisa
        {
            get
            {
                var cache = cop != 0;
                var suld_3d = geom == d3;
                var non_trap = clampm != clamp_trap;
                return cache || suld_3d || non_trap ? HardwareIsa.SM_20 : HardwareIsa.SM_10;
            }
        }

        protected override bool allow_bit8 { get { return true; } }
        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override bool allow_vec { get { return true; } }
        protected override void custom_validate_opcode(Module ctx)
        {
            (geom != 0).AssertTrue();
            (cop == 0 || cop == ca || cop == cg || cop == cs || cop == cv).AssertTrue();
            (dtype.is_bit() && (dtype.is_scalar() || dtype.is_v2() || dtype.is_v4())).AssertTrue();
        }

        public Expression d { get; set; }
        public Expression a { get; set; }
        public Expression b { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, dtype).AssertTrue();
            agree(a, surfref).AssertTrue();
            if (geom == d1) (agree(d, s32) || agree(d, s32.v1)).AssertTrue();
            else if (geom == d2) agree(d, s32.v2).AssertTrue();
            else if (geom == d3) agree(d, s32.v4).AssertTrue();
            else throw AssertionHelper.Fail();
        }
    }
}