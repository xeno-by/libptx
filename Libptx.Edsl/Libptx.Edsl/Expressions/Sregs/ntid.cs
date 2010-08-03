using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Sregs
{
    public class ntid : Libptx.Expressions.Sregs.ntid, sreg
    {
        public static implicit operator v4_u16(ntid ntid) { return new v4_u16(ntid); }
        public static implicit operator v4_s16(ntid ntid) { return new v4_s16(ntid); }
        public static implicit operator v4_b16(ntid ntid) { return new v4_b16(ntid); }
        public static implicit operator v4_u32(ntid ntid) { return new v4_u32(ntid); }
        public static implicit operator v4_s32(ntid ntid) { return new v4_s32(ntid); }
        public static implicit operator v4_b32(ntid ntid) { return new v4_b32(ntid); }
    }
}
