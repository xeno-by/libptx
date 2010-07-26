using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Common.Spaces;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using Type=Libptx.Common.Types.Type;
using XenoGears.Functional;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop("atom{.space}.op.type d, [a], b;")]
    [Ptxop("atom{.space}.op.type d, [a], b, c;")]
    [DebuggerNonUserCode]
    public partial class atom : ptxop
    {
        [Affix] public space space { get; set; }
        [Affix] public op op { get; set; }
        [Affix] public Type type { get; set; }

        protected override HardwareIsa custom_hwisa
        {
            get
            {
                if (space == global) return HardwareIsa.SM_11;
                if (space == shared) return HardwareIsa.SM_12;
                if ((op == add || op == cas || op == exch) && type.is64()) return HardwareIsa.SM_12;
                if (space == shared && type.is64()) return HardwareIsa.SM_20;
                if (op == add && type == f32) return HardwareIsa.SM_20;
                if (space == 0) return HardwareIsa.SM_20;
                return HardwareIsa.SM_10;
            }
        }

        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override void custom_validate_opcode(Module ctx)
        {
            (space == 0 || space == global || space == shared).AssertTrue();
            (op == and || op == or || op == xor || op == cas || op == exch || op == add || op == inc || op == dec || op == min || op == max).AssertTrue();
            (op == and || op == or || op == xor).AssertImplies(type == b32);
            (op == cas || op == exch).AssertImplies(type == b32 || type == b64);
            (op == add).AssertImplies(type == u32 || type == u64 || type == s32 || type == f32);
            (op == inc || op == dec).AssertImplies(type == u32);
            (op == min || op == max).AssertImplies(type == u32 || type == s32 || type == f32);
            (type == b32 || type == b64 || type == u32 || type == u64 || type == s32 || type == f32).AssertTrue();
        }

        public atom() { 1.UpTo(4).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }
        public Expression b { get { return Operands[2]; } set { Operands[2] = value; } }
        public Expression c { get { return Operands[3]; } set { Operands[3] = value; } }

        protected override bool allow_ptr { get { return true; } }
        protected override void custom_validate_operands(Module ctx)
        {
            (agree(d, type) && is_reg(d)).AssertTrue();
            is_ptr(a, space != 0 ? space : (global | shared)).AssertTrue();
            (agree(b, type) && is_reg(b)).AssertTrue();
            (c == null || (agree(c, type) && is_reg(c))).AssertTrue();
            (c != null).AssertImplies(op == cas);
        }
    }
}