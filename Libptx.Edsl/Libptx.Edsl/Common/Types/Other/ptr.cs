using Libptx.Expressions;

namespace Libptx.Edsl.Common.Types.Other
{
    public partial class ptr : typed_expr
    {
        public ptr(Expression expr)
            : base(expr)
        {
        }
    }
}
