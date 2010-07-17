using System;
using System.IO;
using Libptx.Common;

namespace Libptx.Expressions
{
    public class Address : Atom, Expression
    {
        public Addressable Base { get; set; } // may be null
        public Var Offset1 { get; set; } // may be null
        public int Offset2 { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            throw new NotImplementedException();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}