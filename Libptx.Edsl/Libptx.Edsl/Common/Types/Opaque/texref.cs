using Libptx.Expressions;

namespace Libptx.Edsl.Common.Types.Opaque
{
    public partial class texref : typed_expr
    {
        public texref(Expression expr)
            : base(expr)
        {
        }
    }
}
