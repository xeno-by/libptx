using System;
using System.IO;
using Libptx.Common;
using Libptx.Statements;

namespace Libptx.Expressions
{
    public class Address : Atom, Expression
    {
        public Addressable Base { get; set; } // may be null
        public int Offset { get; set; }

        public static implicit operator Address(int offset)
        {
            return new Address { Offset = offset };
        }

        public static implicit operator Address(Var @var)
        {
            return new Address { Base = @var };
        }

        public static implicit operator Address(Label label)
        {
            return new Address { Base = label };
        }

        protected override void CustomValidate(Module ctx)
        {
            throw new NotImplementedException();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            // todo. ld.local.b32 x,[p+-8]; // negative offset
            throw new NotImplementedException();
        }
    }
}