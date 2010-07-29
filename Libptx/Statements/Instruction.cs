using System.Collections.Generic;
using System.Diagnostics;
using Libptx.Common;
using Libptx.Expressions;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Statements
{
    [DebuggerNonUserCode]
    public abstract class Instruction : Atom, Statement
    {
        public Expression Guard { get; set; }

        private IList<Expression> _operands = new List<Expression>();
        public IList<Expression> Operands
        {
            get { return _operands; }
            set { _operands = value ?? new List<Expression>(); }
        }

        protected override void CustomValidate()
        {
            if (Guard != null)
            {
                Guard.Validate();
                Guard.is_pred().AssertTrue();
            }

            Operands.ForEach(arg =>
            {
                // this is commented out because operands may be optional
                // arg.AssertNotNull();

                if (arg != null) arg.Validate();
            });
        }
    }
}