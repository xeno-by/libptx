using System;
using System.IO;
using Libptx.Common;

namespace Libptx.Expressions
{
    public class Index : Atom, Expression
    {
        public Var Base { get; set; } // may be null
        public int Offset { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            throw new NotImplementedException();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            // todo. read up the rules of what is allowed and what is not
            throw new NotImplementedException();
        }
    }
}