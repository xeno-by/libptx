using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Specials
{
    public class tid : Libptx.Expressions.Specials.tid, special
    {
        public static implicit operator v4_u32(tid tid) { return new v4_u32(tid); }
        public static implicit operator v4_s32(tid tid) { return new v4_s32(tid); }
        public static implicit operator v4_b32(tid tid) { return new v4_b32(tid); }
    }
}
