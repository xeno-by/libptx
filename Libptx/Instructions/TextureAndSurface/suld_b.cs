using System;
using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;
using XenoGears.Strings;
using Type = Libptx.Common.Types.Type;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop15("suld.b.geom{.cop}.dtype.clampm    d, [a, b];")]
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
        protected override void custom_validate_opcode()
        {
            (geom != 0).AssertTrue();
            (cop == 0 || cop == ca || cop == cg || cop == cs || cop == cv).AssertTrue();
            (dtype.is_bit() && (dtype.is_scalar() || dtype.is_v2() || dtype.is_v4())).AssertTrue();
        }

        public suld_b() { 1.UpTo(3).ForEach(_ => Operands.Add(null)); }
        [Destination] public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }

        protected override void custom_validate_operands()
        {
            is_reg(d, dtype).AssertTrue();
            (is_surfref(a) || agree(a, u32) || agree(a, u64)).AssertTrue();
            if (geom == d1) (is_alu(d, s32) || is_alu(d, s32.v1)).AssertTrue();
            else if (geom == d2) is_alu(d, s32.v2).AssertTrue();
            else if (geom == d3) is_alu(d, s32.v4).AssertTrue();
            else throw AssertionHelper.Fail();
        }

        protected override string custom_render_ptx(string core)
        {
            var iof = core.IndexOf(",");
            var before = core.Slice(0, iof);
            var after = core.Slice(iof + 2, -1);
            return String.Format("{0}, [{1}];", before, after);
        }
    }
}