using System;
using System.Diagnostics;
using System.IO;
using Libptx.Common;
using Libptx.Common.Spaces;
using Libptx.Expressions.Slots;
using XenoGears.Assertions;

namespace Libptx.Expressions.Addresses
{
    [DebuggerNonUserCode]
    public class Offset : Atom
    {
        public Var Base { get; set; }
        public long Imm { get; set; }

        public static implicit operator Offset(Var @var)
        {
            return @var == null ? null : new Offset { Base = @var };
        }

        public static implicit operator Offset(long offset)
        {
            return new Offset { Imm = offset };
        }

        // an Offset might also be a combo of base var and an immediate constant
        // see Var.Dsl and its user-defined operators for more information

        protected override void CustomValidate(Module ctx)
        {
            if (Base != null)
            {
                (Base.Space == space.reg).AssertTrue();
                Base.Validate(ctx);
            }
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            // todo. +-8
            throw new NotImplementedException();
        }
    }
}