using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Expressions;
using XenoGears.Assertions;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop("bar.sync a{, b};")]
    [DebuggerNonUserCode]
    public partial class bar_sync : ptxop
    {
        protected override SoftwareIsa custom_swisa
        {
            get
            {
                var a_reg = is_reg(a);
                return a_reg || b != null ? SoftwareIsa.PTX_20 : SoftwareIsa.PTX_10;
            }
        }

        protected override HardwareIsa custom_hwisa
        {
            get
            {
                var a_reg = is_reg(a);
                return a_reg || b != null ? HardwareIsa.SM_20 : HardwareIsa.SM_10;
            }
        }

        protected override void custom_validate_opcode(Module ctx)
        {
            // todo. implement the following:
            // Operands a, b, and d have type .u32; operands p and c are predicates.
            // Register operands, thread count ... introduced in PTX ISA version 2.0.
            // Register operands, thread count ... require sm_20 or later.
        }

        public Expression a { get; set; }
        public Expression b { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(a, u32).AssertTrue();
            agree_or_null(b, u32).AssertTrue();

            var a_const = (a as Const).AssertNotNull();
            if (a_const != null)
            {
                var value = @const.AssertCoerce<int>();
                (0 <= value && value <= 15).AssertTrue();
            }
        }
    }
}