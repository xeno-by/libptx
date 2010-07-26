using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Expressions.Immediate;
using Libptx.Instructions.Annotations;
using Libptx.Expressions;
using XenoGears.Assertions;
using XenoGears.Functional;

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

        public bar_sync() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        public Expression a { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression b { get { return Operands[1]; } set { Operands[1] = value; } }

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