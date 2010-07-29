using System;
using System.Diagnostics;
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
        public Type Type { get { return typeof(Bmk); } }

        protected override void CustomValidate()
        {
            Name.ValidateName();
            // uniqueness of names is validated by the context
        }

        protected override void RenderPtx()
        {
            writer.Write(Name);
        }
    }
}