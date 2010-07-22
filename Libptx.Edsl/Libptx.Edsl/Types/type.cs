using System;
using Type=Libptx.Common.Types.Type;
using var=Libptx.Edsl.Vars.var;

namespace Libptx.Edsl.Types
{
    public class type : Type
    {
        public var var { get; set; }
    }
}