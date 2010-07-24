using Libcuda.Versions;
using Libptx.Expressions;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;

namespace Libptx.Instructions.Miscellaneous
{
    [Ptxop("pmevent a;", SoftwareIsa.PTX_14)]
    public partial class pmevent : ptxop
    {
        public Expression a { get; set; }

        protected override void custom_validate_operands(Module ctx)
        {
            var a_const = (a as Const).AssertNotNull();
            var value = a_const.AssertCoerce<int>();
            (0 <= value && value <= 15).AssertTrue();
        }
    }
}