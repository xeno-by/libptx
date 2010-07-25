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

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("mul{.mode}.type             d, a, b;")]
    [Ptxop("mul24{.hi,.lo}.type         d, a, b;")]
    [Ptxop("mul{.rnd}{.ftz}{.sat}.f32   d, a, b;")]
    [Ptxop("mul{.rnd}.f64               d, a, b;")]
    [DebuggerNonUserCode]
    public partial class mul : ptxop
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
                var f32_rmp = type == f32 && (rnd == rm || rn == rp);
                return f32_rmp ? HardwareIsa.SM_20 : HardwareIsa.SM_10;
            }
        }

        protected override void custom_validate_opcode(Module ctx)
        {
            (is24 == true).AssertImplies(type == s32 || type == u32);
            (mode != 0).AssertImplies(type.is_int());
            (mode == wide).AssertImplies(type.is16() || type.is32());
            (rnd != 0).AssertImplies(type.is_float());
            (ftz == true).AssertImplies(type == f32);
            (sat == true).AssertImplies(type == f32);
        }

        mul() { 1.UpTo(3).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, type).AssertTrue();
            agree(a, type).AssertTrue();
            agree(b, type).AssertTrue();
        }
    }
}