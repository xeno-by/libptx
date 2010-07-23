using Libptx.Expressions;

namespace Libptx.Edsl.Common.Types.Opaque
{
    public class samplerref : typed_expr
    {
        public samplerref(Expression expr)
            : base(expr)
        {
        }
    }
}
