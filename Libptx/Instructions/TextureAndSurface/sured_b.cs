using System;
using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;
using XenoGears.Strings;
using Libptx.Expressions;
using XenoGears.Functional;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop20("sured.b.op.geom.ctype.clampm [a, b], c;")]
    [DebuggerNonUserCode]
    public partial class sured_b : ptxop
    {
        [Affix] public op op { get; set; }
        [Affix] public geom geom { get; set; }
        [Affix] public Type ctype { get; set; }
        [Affix] public clampm clampm { get; set; }

        protected override bool allow_bit32 { get { return true; } }
        protected override void custom_validate_opcode()
        {
            (geom != 0).AssertTrue();
            (ctype == u32 || ctype == s32 || ctype == b32 || ctype == u64).AssertTrue();
            (op == add || op == min || op == max || op == or || op == xor).AssertTrue();
            (op == add).AssertImplies(ctype == u32 || ctype == u64 || ctype == s32);
            (op == min || op == max).AssertImplies(ctype == u32 || ctype == s32);
            (op == and || op == or).AssertImplies(ctype == b32);
        }

        public sured_b() { 1.UpTo(3).ForEach(_ => Operands.Add(null)); }
        public Expression a { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression b { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression c { get { return Operands[2]; } set { Operands[2] = value; } }

        protected override void custom_validate_operands()
        {
            (is_surfref(a) || agree(a, u32) || agree(a, u64)).AssertTrue();
            if (geom == d1) (is_alu(b, s32) || is_alu(b, s32.v1)).AssertTrue();
            else if (geom == d2) is_alu(b, s32.v2).AssertTrue();
            else if (geom == d3) is_alu(b, s32.v4).AssertTrue();
            else throw AssertionHelper.Fail();
            is_alu(c, ctype).AssertTrue();
        }

        protected override string custom_render_ptx(string core)
        {
            var iof_comma = core.LastIndexOf(",");
            var big_before = core.Slice(0, iof_comma);
            var iof_whitespace = big_before.IndexOf(" ");
            var opcode = big_before.Slice(0, iof_whitespace);
            var before = big_before.Slice(iof_whitespace + 1);
            var after = core.Slice(iof_comma + 2, -1);
            return String.Format("{0} [{1}], {2};", opcode, before, after);
        }
    }
}