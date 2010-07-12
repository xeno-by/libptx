using System;
using System.Collections.Generic;

namespace Libptx.Expressions
{
    public class Vector
    {
        public List<Var> Vars { get; set; }

        public static implicit operator Vector(Var[] vars)
        {
            throw new NotImplementedException();
        }
    }
}