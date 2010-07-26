using System;
using System.IO;
using Libptx.Common;
using Libptx.Common.Types;
using Libptx.Expressions.Slots;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions.Addresses
{
    public class Index : Atom, Expression
    {
        public Var Base { get; set; } // may be null
        public long Offset { get; set; }

        public Type Type
        {
            get { return typeof(Ptr); }
        }

        protected override void CustomValidate(Module ctx)
        {
            // todo. read up the rules of what is allowed and what is not
            throw new NotImplementedException();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}