using System;
using System.Diagnostics;
using System.IO;
using Libptx.Common;
using Libptx.Common.Spaces;
using Libptx.Common.Types.Pointers;
using Libptx.Expressions.Slots;
using Libptx.Statements;
using Type=Libptx.Common.Types.Type;

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
            // todo. read up the rules of what is allowed and what is not
            throw new NotImplementedException();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            // todo. ld.local.b32 x,[p+-8]; // negative offset
            throw new NotImplementedException();
        }
    }
}