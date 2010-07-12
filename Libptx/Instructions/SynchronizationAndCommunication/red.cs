using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop10("red{.space}.op.type [a], b;", SoftwareIsa.PTX_12)]
    internal class red : ptxop
    {
        [Infix] public ss space { get; set; }
        [Infix] public op op { get; set; }
        [Infix] public type type { get; set; }

        protected override HardwareIsa custom_hwisa
        {
            get
            {
                if (space == global) return HardwareIsa.SM_11;
                if (space == shared) return HardwareIsa.SM_12;
                if (op == add && type.is64()) return HardwareIsa.SM_12;
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
            (op == and || op == or || op == xor || op == add || op == inc || op == dec).AssertTrue();
            (op == and || op == or || op == xor).AssertImplies(type == b32);
            (op == add).AssertImplies(type == u32 || type == u64 || type == s32 || type == f32);
            (op == inc || op == dec).AssertImplies(type == u32);
            (op == min || op == max).AssertImplies(type == u32 || type == s32 || type == f32);
            (type == b32 || type == b64 || type == u32 || type == u64 || type == s32 || type == f32).AssertTrue();

            // todo. implement this:
            // Operand a must reside in either the global or shared state space.
        }
    }
}