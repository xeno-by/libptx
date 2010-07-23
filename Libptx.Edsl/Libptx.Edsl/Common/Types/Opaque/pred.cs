using Libptx.Expressions;

namespace Libptx.Edsl.Common.Types.Opaque
{
    public class pred : typed_expr
    {
        public pred(Expression expr)
            : base(expr)
        {
        }
    }
}
