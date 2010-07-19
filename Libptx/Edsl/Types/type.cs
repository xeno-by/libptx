using Libptx.Common.Types;
using var=Libptx.Edsl.Vars.var;

namespace Libptx.Edsl.Types
{
    public class type : Type
    {
        // todo. verify type of the var
        public var var { get; set; }
    }
}