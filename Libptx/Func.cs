using System.Collections.Generic;
using Libptx.Expressions;
using Libptx.Statements;

namespace Libptx
{
    public class Func : Block, Callable
    {
        public IList<Var> Params { get; private set; }
        public IList<Var> Rets { get; private set; }
    }
}