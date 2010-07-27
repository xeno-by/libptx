using System;
using System.Diagnostics;
using System.IO;
using Libptx.Common;
using Libptx.Expressions.Slots;
using XenoGears.Assertions;

namespace Libptx.Expressions.Addresses
{
    [DebuggerNonUserCode]
    public partial class Offset : Atom
    {
        public Expression Base { get; set; }
        public long Imm { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            if (Base != null)
            {
                Base.Validate(ctx);
                (Base is Reg || Base is Var).AssertTrue();

                var base_reg = Base as Reg;
                if (base_reg != null)
                {
                    (agree(base_reg.Type, u32) || agree(base_reg.Type, u64)).AssertTrue();
                }

                var base_var = Base as Var;
                if (base_var != null)
                {
                    base_var.is_opaque().AssertFalse();
                }
            }
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            // todo. +-8
            throw new NotImplementedException();
        }
    }
}