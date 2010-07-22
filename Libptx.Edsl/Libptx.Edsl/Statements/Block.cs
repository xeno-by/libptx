using System.Collections.Generic;
using Libptx.Expressions;
using Libptx.Statements;

namespace Libptx.Edsl
{
    public static partial class Ptx21
    {
        public static partial class Sm20
        {
            public class Block : Libptx.Statements.Block
            {
                private readonly Libptx.Statements.Block _base;
                public Block() : this(new Libptx.Statements.Block()) { }
                internal Block(Libptx.Statements.Block @base) { _base = @base; }

                public override IList<Var> Vars
                {
                    get { return _base.Vars; }
                    set { base.Vars = value; }
                }

                public override IList<Statement> Stmts
                {
                    get { return _base.Stmts; }
                    set { base.Stmts = value; }
                }
            }
        }
    }
}
