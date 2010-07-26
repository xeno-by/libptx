using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Libptx.Common;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions
{
    [DebuggerNonUserCode]
    public class Modded : Atom, Expression
    {
        public Expression Expr { get; set; }
        public Mod Mod { get; set; }

        private IList<Expression> _embedded = new List<Expression>();
        public IList<Expression> Embedded
        {
            get { return _embedded; }
            set { _embedded = value ?? new List<Expression>(); }
        }

        public Type Type
        {
            get { throw new NotImplementedException(); }
        }

        protected override void CustomValidate(Module ctx)
        {
            // todo. what expressions can be modded?
            // e.g. does PTX allow modding immediate expressions?
            // couple can consist only of two preds and not may be applied only to s32
            throw new NotImplementedException();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}