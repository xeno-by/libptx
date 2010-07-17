using System.Collections.Generic;
using Libptx.Expressions;

namespace Libptx
{
    public interface Callable : Addressable
    {
        IList<Var> Params { get; set; }
        IList<Var> Rets { get; set; }
    }
}