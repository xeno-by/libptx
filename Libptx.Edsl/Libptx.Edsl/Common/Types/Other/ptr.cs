using Libptx.Expressions;

namespace Libptx.Edsl.Common.Types.Other
{
    public class ptr : typed_expr
    {
        public ptr(Expression expr)
            : base(expr)
        {
        }
    }
}