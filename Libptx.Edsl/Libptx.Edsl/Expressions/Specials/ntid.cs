using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Specials
{
    public class ntid : Libptx.Expressions.Specials.ntid, special
    {
        public static implicit operator v4_u32(ntid ntid) { return new v4_u32(ntid); }
        public static implicit operator v4_s32(ntid ntid) { return new v4_s32(ntid); }
        public static implicit operator v4_b32(ntid ntid) { return new v4_b32(ntid); }
    }
}
