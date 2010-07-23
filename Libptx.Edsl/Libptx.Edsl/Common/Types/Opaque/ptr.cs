using Libptx.Expressions;

namespace Libptx.Edsl.Common.Types.Opaque
{
    public class ptr : typed_expr
    {
        public ptr(Expression expr)
            : base(expr)
        {
        }
    }
}
