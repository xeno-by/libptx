using Libptx.Expressions;

namespace Libptx.Edsl.Common.Types.Opaque
{
    public class surfref : typed_expr
    {
        public surfref(Expression expr)
            : base(expr)
        {
        }
    }
}
