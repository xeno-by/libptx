using System;
using System.Collections.Generic;
using System.IO;
using Libptx.Common;
using Libptx.Expressions.Slots;
using XenoGears.Functional;

namespace Libptx.Statements
{
    public class Block : Atom, Statement
    {
        private IList<Var> _vars = new List<Var>();
        public virtual IList<Var> Vars
        {
            get { return _vars; }
            set { _vars = value ?? new List<Var>(); }
        }

        private IList<Statement> _stmts = new List<Statement>();
        public virtual IList<Statement> Stmts
        {
            get { return _stmts; }
            set { _stmts = value ?? new List<Statement>(); }
        }

        protected override void CustomValidate(Module ctx)
        {
            Vars.ForEach(@var => @var.Validate(ctx));
            Stmts.ForEach(stmt => stmt.Validate(ctx));
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}