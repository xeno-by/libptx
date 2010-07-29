using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Expressions.Immediate;
using Libptx.Expressions.Slots;
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
            get { return a is Reg ? SoftwareIsa.PTX_20 : SoftwareIsa.PTX_10; }
        }

        protected override HardwareIsa custom_hwisa
        {
            get { return a is Reg ? HardwareIsa.SM_20 : HardwareIsa.SM_10; }
        }

        public bar_sync() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        public Expression a { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression b { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands()
        {
            is_alu(a, u32).AssertTrue();
            is_alu_or_null(b, u32).AssertTrue();

            var a_const = (a as Const).AssertNotNull();
            if (a_const != null)
            {
                var value = a_const.AssertCoerce<int>();
                (0 <= value && value <= 15).AssertTrue();
            }
        }
    }
}