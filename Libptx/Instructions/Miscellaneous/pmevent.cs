using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Expressions;
using Libptx.Expressions.Immediate;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Instructions.Miscellaneous
{
    [Ptxop("pmevent a;", SoftwareIsa.PTX_14)]
    [DebuggerNonUserCode]
    public partial class pmevent : ptxop
    {
        public pmevent() { 1.UpTo(1).ForEach(_ => Operands.Add(null)); }
        public Expression a { get { return Operands[0]; } set { Operands[0] = value; } }

        protected override void custom_validate_operands()
        {
            var a_const = (a as Const).AssertNotNull();
            var value = a_const.AssertCoerce<int>();
            (0 <= value && value <= 15).AssertTrue();
        }
    }
}