using System;
using System.IO;
using Libptx.Common;
using Libptx.Expressions.Slots;
using Libptx.Statements;

namespace Libptx.Expressions.Addresses
{
    public class Offset : Renderable, Validatable
    {
        public Addressable Base { get; set; }
        public long Imm { get; set; }

        public static implicit operator Offset(Var @var)
        {
            return @var == null ? null : new Offset { Base = @var };
        }

        public static implicit operator Offset(Label label)
        {
            return label == null ? null : new Offset { Base = label };
        }

        public static implicit operator Offset(long offset)
        {
            return new Offset { Imm = offset };
        }

        public void Validate(Module ctx)
        {
            // todo. read up the rules of what is allowed and what is not
            throw new NotImplementedException();
        }

        public void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}