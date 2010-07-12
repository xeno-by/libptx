using System;
using System.Collections.Generic;
using Libptx.Common.Infrastructure;

namespace Libptx.Expressions
{
    public class Vector : Atom, Expression
    {
        public IList<Var> Vars { get; private set; }

        public static implicit operator Vector(Var[] vars)
        {
            throw new NotImplementedException();
        }
    }
}