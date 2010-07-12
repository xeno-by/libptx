using System.Collections.Generic;
using Libptx.Expressions;

namespace Libptx.Statements
{
    public class Block : Statement
    {
        public List<Var> Vars { get; set; }
        public List<Statement> Stmts { get; set; }
    }
}