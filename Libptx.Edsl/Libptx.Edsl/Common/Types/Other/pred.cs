using Libptx.Expressions;

namespace Libptx.Edsl.Common.Types.Other
{
    public partial class pred : typed_expr
    {
        public pred(Expression expr)
            : base(expr)
        {
        }
    }
}
