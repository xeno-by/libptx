using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("sub.type                    d, a, b;")]
    [Ptxop("sub{.sat}.s32               d, a, b;")]
    [Ptxop("sub.cc.type                 d, a, b;")]
    [Ptxop("subc{.cc}.type              d, a, b;")]
    [Ptxop("sub{.rnd}{.ftz}{.sat}.f32   d, a, b;")]
    [Ptxop("sub{.rnd}.f64               d, a, b;")]
    [DebuggerNonUserCode]
    public partial class sub : ptxop
    {
        [Mod(SoftwareIsa.PTX_13)] public bool c { get; set; }
        [Affix(SoftwareIsa.PTX_13)] public bool cc { get; set; }
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
            (c || cc).AssertImplies(type == s32 || type == u32);
            (c || cc).AssertImplies(sat == false);
            (rnd != 0).AssertImplies(type.is_float());
            (ftz == true).AssertImplies(type == f32);
            (sat == true).AssertImplies(type == s32 || type == f32);
        }

        sub() { 1.UpTo(3).ForEach(_ => Operands.Add(null)); }
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