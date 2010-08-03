using Libptx.Expressions.Immediate;

namespace Libptx.Edsl.Common.Types.Scalar
{
    public partial class reg_u32
    {
        public static implicit operator reg_u32(WarpSz warpSz)
        {
            return new reg_u32(warpSz);
        }
    }
}
