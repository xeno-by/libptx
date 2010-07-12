using System.Collections.Generic;
using Libptx.Expressions;

namespace Libptx
{
    public interface Callable : Addressable
    {
        List<Var> Params { get; set; }
        List<Var> Rets { get; set; }
    }
}