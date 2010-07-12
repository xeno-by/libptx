using System.Collections.Generic;
using Libptx.Common.Infrastructure;
using Libptx.Expressions;

namespace Libptx.Statements
{
    public abstract class Instruction : Atom, Statement
    {
        public virtual Var Guard { get; set; } // may be null and may have the "Not" varmod
        public IList<Expression> Operands { get; private set; }
    }
}