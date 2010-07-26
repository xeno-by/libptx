using System.Linq;

namespace Libptx.Expressions.Immediate
{
    public partial class Vector
    {
        public static implicit operator Vector(Expression[] exprs)
        {
            return exprs == null ? null : new Vector{Elements = exprs.ToList()};
        }
    }
}