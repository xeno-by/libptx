using var=Libptx.Edsl.Vars.var;

namespace Libptx.Edsl.Types.Relaxed
{
    public class relaxed_type<T>
        where T : type
    {
        // todo. verify type of the var
        public var var { get; set; }
    }
}