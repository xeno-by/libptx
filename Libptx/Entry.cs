using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Common.Types;
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

        private Block _body = new Block();
        public virtual Block Body
        {
            get { return _body; }
            set { _body = value ?? new Block(); }
        }

        protected override void CustomValidate(Module ctx)
        {
            Params.AssertEach(p => p.Space == param);

            var size_limit = 256;
            if (ctx.Version >= SoftwareIsa.PTX_15) size_limit += 4096;
            // todo. what if we pass pointers? what if we pass textures?
            (Params.Sum(p => p.SizeInMemory()) <= size_limit).AssertTrue();

            // 5.3. Texture, Sampler, and Surface Types
            // The use of these opaque types is limited to:
            // * Variable definition within global (module) scope and in kernel entry parameter lists.
            // * Static initialization of module-scope variables using comma-delimited static
            //   assignment expressions for the named members of the type
            Params.ForEach(p => p.Type.is_opaque().AssertImplies(p.Init == null));

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