using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Libptx.Common;
using Libptx.Expressions.Slots;
using XenoGears.Functional;
using Libptx.Common.Types;
using XenoGears.Assertions;

namespace Libptx.Statements
{
    [DebuggerNonUserCode]
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
            // 5.3. Texture, Sampler, and Surface Types
            // The use of these opaque types is limited to:
            // * Variable definition within global (module) scope and in kernel entry parameter lists.
            // * Static initialization of module-scope variables using comma-delimited static
            //   assignment expressions for the named members of the type
            Vars.ForEach(@var => @var.Type.is_opaque().AssertFalse());

            Vars.ForEach(@var => @var.Validate(ctx));
            Stmts.ForEach(stmt => stmt.Validate(ctx));
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}