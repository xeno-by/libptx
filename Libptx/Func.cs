using System.Collections.Generic;
using Libptx.Expressions;
using Libptx.Statements;

namespace Libptx
{
    public class Func : Block, Callable
    {
        public List<Var> Params { get; set; }
        public List<Var> Rets { get; set; }
    }
}