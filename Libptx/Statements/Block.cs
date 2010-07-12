using System.Collections.Generic;
using Libptx.Common.Infrastructure;
using Libptx.Expressions;
using XenoGears.Functional;
using XenoGears.Collections.Observable;

namespace Libptx.Statements
{
    public class Block : Atom, Statement
    {
        public IList<Var> Vars { get; private set; }
        public IList<Statement> Stmts { get; private set; }

        public Block()
        {
            Vars = new List<Var>().Observe();
            Stmts = new List<Statement>().Observe();
        }

        public override void Validate()
        {
            Vars.ForEach(@var => @var.Validate());
            Stmts.ForEach(stmt => stmt.Validate());
        }
    }
}