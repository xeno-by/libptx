using System;
using System.Collections.Generic;
using Libptx.Common.Comments;
using Libptx.Common.Performance;
using Libptx.Common.Performance.Pragmas;
using Libptx.Functions;
using Libptx.Statements;

namespace Libptx.Edsl.Functions
{
    public class Entry : Libptx.Functions.Entry
    {
        private readonly Libptx.Functions.Entry _base;
        public Entry() : this(new Libptx.Functions.Entry()) { }
        internal Entry(Libptx.Functions.Entry @base) { _base = @base; }

        public override IList<Comment> Comments
        {
            get { return _base.Comments; }
            set { _base.Comments = value; }
        }

        public override IList<Pragma> Pragmas
        {
            get { return _base.Pragmas; }
            set { _base.Pragmas = value; }
        }

        public override String Name
        {
            get { return _base.Name; }
            set { _base.Name = value; }
        }

        public override Tuning Tuning
        {
            get { return _base.Tuning; }
            set { _base.Tuning = value; }
        }

        public override Params Params
        {
            get { return _base.Params; }
            set { _base.Params = value; }
        }

        public override IList<Statement> Stmts
        {
            get { return _base.Stmts; }
            set { _base.Stmts = value; }
        }
    }
}