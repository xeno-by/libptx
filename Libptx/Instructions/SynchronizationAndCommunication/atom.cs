using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop("atom{.space}.op.type d, [a], b;")]
    [Ptxop("atom{.space}.op.type d, [a], b, c;")]
    internal class atom : ptxop
    {
        [Suffix] public ss space { get; set; }
        [Suffix] public op op { get; set; }
        [Suffix] public type type { get; set; }

        protected override HardwareIsa custom_hwisa
        {
            get
            {
                if (space == global) return HardwareIsa.SM_11;
                if (space == shared) return HardwareIsa.SM_12;
                if ((op == add || op == cas || op == exch) && type.is64()) return HardwareIsa.SM_12;
                if (space == shared && type.is64()) return HardwareIsa.SM_20;
                if (op == add && type == f32) return HardwareIsa.SM_20;
                if (space == null) return HardwareIsa.SM_20;
                return HardwareIsa.SM_10;
            }
        }

        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (space == null || space == global || space == shared).AssertTrue();
            (op == and || op == or || op == xor || op == cas || op == exch || op == add || op == inc || op == dec || op == min || op == max).AssertTrue();
            (op == and || op == or || op == xor).AssertImplies(type == b32);
            (op == cas || op == exch).AssertImplies(type == b32 || type == b64);
            (op == add).AssertImplies(type == u32 || type == u64 || type == s32 || type == f32);
            (op == inc || op == dec).AssertImplies(type == u32);
            (op == min || op == max).AssertImplies(type == u32 || type == s32 || type == f32);
            (type == b32 || type == b64 || type == u32 || type == u64 || type == s32 || type == f32).AssertTrue();

            // todo. implement this:
            // Operand a must reside in either the global or shared state space.
        }
    }
}