using Libptx.Expressions;

namespace Libptx.Edsl.Common.Types
{
    public class typed_expr
    {
        public Expression expr { get; private set; }

        public typed_expr(Expression expr)
        {
            this.expr = expr;
        }
    }
}
