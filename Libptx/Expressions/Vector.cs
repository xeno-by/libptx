using System;
using System.Collections.Generic;
using System.IO;
using Libptx.Common;

namespace Libptx.Expressions
{
    public class Vector : Atom, Expression
    {
        private IList<Var> _vars = new List<Var>();
        public IList<Var> Vars
        {
            get { return _vars; }
            set { _vars = value ?? new List<Var>(); }
        }

        public static implicit operator Vector(Var[] vars)
        {
            throw new NotImplementedException();
        }

        protected override void CustomValidate(Module ctx)
        {
            throw new NotImplementedException();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}