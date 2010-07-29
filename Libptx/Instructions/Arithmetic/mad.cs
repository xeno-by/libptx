using System;
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
using XenoGears.Strings;

namespace Libptx.Instructions.Arithmetic
{
    [Ptxop("mad{.hi,.lo,.wide}.type     d, a, b, c;")]
    [Ptxop("mad.hi.sat.s32              d, a, b, c;")]
    [Ptxop("mad24{.hi,.lo}.type         d, a, b, c;")]
    [Ptxop("mad24.hi.sat.s32            d, a, b, c;")]
    [Ptxop("mad{.ftz}{.sat}.f32         d, a, b, c;")]
    [Ptxop("mad.rnd{.ftz}{.sat}.f32     d, a, b, c;")]
    [Ptxop("mad.rnd.f64                 d, a, b, c;")]
    [DebuggerNonUserCode]
    public partial class mad : ptxop
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
                var f32_rnd = type == f32 && rnd != 0;
                return f32_rnd ? HardwareIsa.SM_20 : HardwareIsa.SM_10;
            }
        }

        protected override void custom_validate_opcode()
        {
            (is24 == true).AssertImplies(type == s32 || type == u32);
            (mode != 0).AssertImplies(type.is_int());
            (mode == wide).AssertImplies(type.is16() || type.is32());
            (rnd != 0).AssertImplies(type.is_float());
            (type == f64).AssertImplies(rnd != 0);
            (ftz == true).AssertImplies(type == f32);
            (sat == true).AssertImplies(type == s32 || type == f32);
            (sat == true && type.is_int()).AssertImplies(mode == mulm_hi);

            (ctx.Version >= SoftwareIsa.PTX_14 && type == f64).AssertImplies(rnd != 0);
            (ctx.Target >= HardwareIsa.SM_20 && type == f32).AssertImplies(rnd != 0);
        }

        public mad() { 1.UpTo(4).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }
        public Expression c { get { return Operands[3]; } set { Operands[3] = value; } }

        protected override void custom_validate_operands()
        {
            if (mode == wide)
            {
                var s_typename = type.Name.ToString();
                var s_twice_typename = s_typename.Slice(0, 1) + int.Parse(s_typename.Slice(1)) * 2;
                var twice_typename = (TypeName)Enum.Parse(typeof(TypeName), s_twice_typename);
                var twice_type = new Type { Name = twice_typename, Mod = type.Mod, Dims = type.Dims };

                is_alu(d, twice_type).AssertTrue();
                is_alu(a, type).AssertTrue();
                is_alu(b, type).AssertTrue();
                is_alu(c, twice_type).AssertTrue();
            }
            else
            {
                is_reg(d, type).AssertTrue();
                is_alu(a, type).AssertTrue();
                is_alu(b, type).AssertTrue();
                is_alu(c, type).AssertTrue();
            }
        }
    }
}