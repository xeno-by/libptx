using System.Collections.Generic;
using Libptx.Expressions;

namespace Libptx.Statements
{
    public abstract class Instruction : Statement
    {
        public Var Guard { get; set; } // may be null and may have the "Not" varmod
        public List<Expression> Operands { get; set; } // it'd better be observable
    }
}