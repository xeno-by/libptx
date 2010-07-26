using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;
using Type = Libptx.Common.Types.Type;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop15("sust.b.geom{.cop}.ctype.clampm    [a, b], c;")]
    [DebuggerNonUserCode]
    public partial class sust_b : ptxop
    {
        [Affix] public geom geom { get; set; }
        [Affix] public cop cop { get; set; }
        [Affix] public Type ctype { get; set; }
        [Affix] public clampm clampm { get; set; }

        protected override SoftwareIsa custom_swisa
        {
            get
            {
                var cache = cop != 0;
                var sust_3d = geom == d3;
                var non_trap = clampm != clamp_trap;
                return cache || sust_3d || non_trap ? SoftwareIsa.PTX_20 : SoftwareIsa.PTX_15;
            }
        }

        protected override HardwareIsa custom_hwisa
        {
            get
            {
                var cache = cop != 0;
                var sust_3d = geom == d3;
                var non_trap = clampm != clamp_trap;
                return cache || sust_3d || non_trap ? HardwareIsa.SM_20 : HardwareIsa.SM_10;
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
            (cop == 0 || cop == wb || cop == cg || cop == cs || cop == wt).AssertTrue();
            (ctype.is_bit() && (ctype.is_scalar() || ctype.is_v2() || ctype.is_v4())).AssertTrue();
        }

        public sust_b() { 1.UpTo(3).ForEach(_ => Operands.Add(null)); }
        public Expression a { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression b { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression c { get { return Operands[2]; } set { Operands[2] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(a, surfref).AssertTrue();
            if (geom == d1) (agree(b, s32) || agree(b, s32.v1)).AssertTrue();
            else if (geom == d2) agree(b, s32.v2).AssertTrue();
            else if (geom == d3) agree(b, s32.v4).AssertTrue();
            else throw AssertionHelper.Fail();
            agree(c, ctype).AssertTrue();
        }
    }
}