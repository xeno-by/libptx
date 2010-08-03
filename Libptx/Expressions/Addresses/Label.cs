using System;
using System.Diagnostics;
using Libptx.Common;
using Libptx.Common.Types.Pointers;
using Libptx.Statements;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions.Addresses
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

        protected override void RenderCubin()
        {
            throw new NotImplementedException();
        }
    }
}