using System;
using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Expressions;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;

namespace Libptx.Instructions.ControlFlow
{
    [Ptxop("bra{.uni} tgt;")]
    [DebuggerNonUserCode]
    public partial class bra : ptxop
    {
        [Affix] public new bool uni { get; set; }

        public Expression tgt { get; set; }

        protected override bool allow_ptr { get { return true; } }
        protected override void custom_validate_operands(Module ctx)
        {
            is_ptr(tgt, code).AssertTrue();
        }
    }
}
