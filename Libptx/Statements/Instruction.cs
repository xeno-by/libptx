using System;
using System.Collections.Generic;
using Libptx.Common;
using Libptx.Expressions;

namespace Libptx.Statements
{
    public abstract class Instruction : Atom, Statement
    {
        public Expression Guard { get; set; } // may be null

        private IList<Expression> _operands = new List<Expression>();
        public IList<Expression> Operands
        {
            get { return _operands; }
            set { _operands = value ?? new List<Expression>(); }
        }

        protected override void CustomValidate(Module ctx)
        {
            // todo. validate guard => only var or?!
            // todo. also validate its type
            throw new NotImplementedException();
        }
    }
}