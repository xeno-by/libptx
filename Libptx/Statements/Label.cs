using System;
using System.Diagnostics;
using System.IO;
using Libptx.Common;
using Libptx.Common.Types.Pointers;
using Libptx.Expressions;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Statements
{
    [DebuggerNonUserCode]
    public class Label : Atom, Expression, Statement
    {
        public String Name { get; set; }
        public Type Type { get { return typeof(Ptr); } }

        protected override void CustomValidate(Module ctx)
        {
            if (Name != null) Name.ValidateName();
            // uniqueness of names is validated by the scope
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}