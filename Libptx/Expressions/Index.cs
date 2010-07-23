using System;
using System.IO;
using Libptx.Common.Types;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions
{
    public class Index : Expression
    {
        public Var Base { get; set; } // may be null
        public int Offset { get; set; }

        public override Type Type
        {
            get { return typeof(Ptr); }
        }

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