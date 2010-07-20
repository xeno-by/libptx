using System.Collections.Generic;
using Libptx.Common;
using Libptx.Expressions;

namespace Libptx.Statements
{
    public abstract class Instruction : Atom, Statement
    {
        // todo. override this in concrete instructions with var_pred/var_couple
        public Var Guard { get; set; } // may be null

        private IList<Expression> _operands = new List<Expression>();
        public IList<Expression> Operands
        {
            get { return _operands; }
            set { _operands = value ?? new List<Expression>(); }
        }
    }
}