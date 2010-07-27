using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Expressions;
using Libptx.Expressions.Addresses;
using Libptx.Instructions.Annotations;
using Libptx.Statements;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Instructions.ControlFlow
{
    [Ptxop("bra{.uni} tgt;")]
    [DebuggerNonUserCode]
    public partial class bra : ptxop
    {
        [Affix] public new bool uni { get; set; }

        public bra() { 1.UpTo(1).ForEach(_ => Operands.Add(null)); }
        public Expression tgt { get { return Operands[0]; } set { Operands[0] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            is_bmk(tgt).AssertTrue();
        }
    }
}
