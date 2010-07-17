using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Expressions;
using Libptx.Statements;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx
{
    public class Entry : Atom
    {
        public String Name { get; set; }

        private Tuning _tuning = new Tuning();
        public Tuning Tuning
        {
            get { return _tuning; }
            set { _tuning = value ?? new Tuning(); }
        }

        private IList<Var> _params = new List<Var>();
        public IList<Var> Params
        {
            get { return _params; }
            set { _params = value ?? new List<Var>(); }
        }

        private Block _body = new Block();
        public Block Body
        {
            get { return _body; }
            set { _body = value ?? new Block(); }
        }

        protected override void CustomValidate(Module ctx)
        {
            Params.AssertEach(p => p.Space == param);

            var size_limit = 256;
            if (ctx.Version >= SoftwareIsa.PTX_15) size_limit += 4096;
            (Params.Sum(p => Marshal.SizeOf(p.Type)) <= size_limit).AssertTrue();

            Tuning.Validate(ctx);
            Params.ForEach(p => p.Validate(ctx));
            Body.Validate(ctx);
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            // todo. also implement this:
            // For PTX ISA version 1.4 and later, parameter variables are declared in the kernel
            // parameter list. For PTX ISA versions 1.0 through 1.3, parameter variables are
            // declared in the kernel body.

            throw new NotImplementedException();
        }
    }
}