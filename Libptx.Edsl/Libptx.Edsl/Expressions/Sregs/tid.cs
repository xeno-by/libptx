using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Sregs
{
    public class tid : Libptx.Expressions.Sregs.tid, sreg
    {
        public static implicit operator v4_u16(tid tid) { return new v4_u16(tid); }
        public static implicit operator v4_s16(tid tid) { return new v4_s16(tid); }
        public static implicit operator v4_b16(tid tid) { return new v4_b16(tid); }
        public static implicit operator v4_u32(tid tid) { return new v4_u32(tid); }
        public static implicit operator v4_s32(tid tid) { return new v4_s32(tid); }
        public static implicit operator v4_b32(tid tid) { return new v4_b32(tid); }
    }
}
