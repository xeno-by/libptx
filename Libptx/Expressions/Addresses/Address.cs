using System;
using System.Diagnostics;
using System.IO;
using Libptx.Common;
using Libptx.Common.Spaces;
using Libptx.Common.Types.Pointers;
using Libptx.Expressions.Slots;
using Libptx.Statements;
using Type=Libptx.Common.Types.Type;
using XenoGears.Assertions;

namespace Libptx.Expressions.Addresses
{
    [DebuggerNonUserCode]
    public class Address : Atom, Expression
    {
        public Addressable Base { get; set; } // may be null
        public Offset Offset { get; set; }

        public Type Type
        {
            get { return typeof(Ptr); }
        }

        public static implicit operator Address(long offset)
        {
            return new Address { Offset = offset };
        }

        public static implicit operator Address(Offset offset)
        {
            return offset == null ? null : new Address { Offset = offset };
        }

        public static implicit operator Address(Var @var)
        {
            return @var == null ? null : @var.Space == space.reg ? new Address { Offset = @var } : new Address { Base = @var };
        }

        public static implicit operator Address(Label label)
        {
            return label == null ? null : new Address { Base = label };
        }

        protected override void CustomValidate(Module ctx)
        {
            (Base != null || Offset != null).AssertTrue();
            if (Base != null) Base.Validate(ctx);
            if (Offset != null) Offset.Validate(ctx);

            // the following combos of Base and Offset are valid:
            //
            //         |      null     |   null + off  |   reg + off   | non-reg + off |
            // ========|===============|===============|===============|===============|
            // null    |    invalid    |      [42]     |    [A + 42]   |    invalid    |
            // reg     |    invalid    |    invalid    |    invalid    |    invalid    |
            // non-reg |      [A]      |      A[42]    |    A[B + 42]  |    invalid    |
            // label   |      lbl      |    invalid    |    invalid    |    invalid    |
            // ========|===============|===============|===============|===============|

            if (Base != null)
            {
                var base_var = Base as Var;
                if (base_var != null)
                {
                    (base_var.Space != reg).AssertTrue();
                }
                else
                {
                    (Offset == null).AssertTrue();
                }
            }
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            // todo. ld.local.b32 x,[p+-8]; // negative offset
            throw new NotImplementedException();
        }
    }
}