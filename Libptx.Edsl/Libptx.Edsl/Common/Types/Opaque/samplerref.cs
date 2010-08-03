using Libptx.Expressions;

namespace Libptx.Edsl.Common.Types.Opaque
{
    public partial class samplerref : typed_expr
    {
        public samplerref(Expression expr)
            : base(expr)
        {
        }
    }
}
