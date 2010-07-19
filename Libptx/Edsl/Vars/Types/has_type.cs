using Libptx.Edsl.Types;

namespace Libptx.Edsl.Vars.Types
{
    public class has_type<T> : var
        where T : type
    {
    }
}