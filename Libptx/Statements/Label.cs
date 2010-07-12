using System;
using Libptx.Common.Infrastructure;
using Libptx.Expressions;

namespace Libptx.Statements
{
    public class Label : Atom, Expression, Statement
    {
        public virtual String Name { get; set; } // may be null
    }
}