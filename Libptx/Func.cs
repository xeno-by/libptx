using System;
using System.Collections.Generic;
using System.IO;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Expressions;
using Libptx.Statements;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx
{
    public class Func : Atom, Callable
    {
        public String Name { get; set; }

        private IList<Var> _params = new List<Var>();
        public IList<Var> Params
        {
            get { return _params; }
            set { _params = value ?? new List<Var>(); }
        }

        private IList<Var> _rets = new List<Var>();
        public IList<Var> Rets
        {
            get { return _rets; }
            set { _rets = value ?? new List<Var>(); }
        }

        private Block _body = new Block();
        public Block Body
        {
            get { return _body; }
            set { _body = value ?? new Block(); }
        }

        protected override void CustomValidate(Module ctx)
        {
            var abi2 = ctx.Target >= HardwareIsa.SM_20 && ctx.Version >= SoftwareIsa.PTX_20;
            Params.AssertEach(p => p.Space == reg || (abi2 && p.Space == param));

            Params.ForEach(p => p.Validate(ctx));
            Rets.ForEach(p => p.Validate(ctx));
            Body.Validate(ctx);
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}