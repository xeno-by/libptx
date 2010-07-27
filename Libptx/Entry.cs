using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Common.Performance;
using Libptx.Expressions.Slots;
using Libptx.Statements;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx
{
    [DebuggerNonUserCode]
    public class Entry : Atom
    {
        private String _name = null;
        public virtual String Name
        {
            get { return _name; }
            set { _name = value; }
        }

        private Tuning _tuning = new Tuning();
        public virtual Tuning Tuning
        {
            get { return _tuning; }
            set { _tuning = value ?? new Tuning(); }
        }

        private Params _params = new Params();
        public virtual Params Params
        {
            get { return _params; }
            set { _params = value ?? new Params(); }
        }

        private IList<Statement> _stmts = new List<Statement>();
        public virtual IList<Statement> Stmts
        {
            get { return _stmts; }
            set { _stmts = value ?? new List<Statement>(); }
        }

        protected override void CustomValidate(Module ctx)
        {
            // if name is null, it'll be autogenerated
            if (Name != null) Name.ValidateName();
            // uniqueness of names is validated by the scope

            Tuning.Validate(ctx);

            var size_limit = 256;
            if (ctx.Version >= SoftwareIsa.PTX_15) size_limit += 4096;
            // opaque types don't count against parameter list size limit
            (Params.Sum(p => p.SizeInMemory()) <= size_limit).AssertTrue();

            Params.ForEach(p =>
            {
                p.AssertNotNull();
                p.Validate(ctx);
                (p.Space == param).AssertTrue();
            });

            Stmts.ForEach(stmt =>
            {
                stmt.AssertNotNull();
                stmt.Validate(ctx);

                var lbl = stmt as Label;
                if (lbl != null && lbl.Name != null)
                {
                    (Stmts.OfType<Label>().Count(lbl2 => lbl2.Name == lbl.Name) == 1).AssertTrue();
                }
            });
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            // todo. also implement this:
            // For PTX ISA version 1.4 and later, parameter variables are declared in the kernel
            // parameter list. For PTX ISA versions 1.0 through 1.3, parameter variables are
            // declared in the kernel body.

            // todo. also implement this:
            // texrefs and samplerrefs might only be param or global
            // in the latter case they're module-wide

            // todo. backwards compatibility for textures:
            // render .global .texref as .tex .u32
            // render tex instruction without brackets

            // todo. also correctly render pragmas

            throw new NotImplementedException();
        }
    }
}