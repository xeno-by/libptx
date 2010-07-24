using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Specials
{
    public class nctaid : Libptx.Expressions.Specials.nctaid, special
    {
        public static implicit operator v4_u16(nctaid nctaid) { return new v4_u16(nctaid); }
        public static implicit operator v4_s16(nctaid nctaid) { return new v4_s16(nctaid); }
        public static implicit operator v4_b16(nctaid nctaid) { return new v4_b16(nctaid); }
        public static implicit operator v4_u32(nctaid nctaid) { return new v4_u32(nctaid); }
        public static implicit operator v4_s32(nctaid nctaid) { return new v4_s32(nctaid); }
        public static implicit operator v4_b32(nctaid nctaid) { return new v4_b32(nctaid); }
    }
}
