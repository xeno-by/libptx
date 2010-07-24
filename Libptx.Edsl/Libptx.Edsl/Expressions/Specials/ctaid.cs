using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Specials
{
    public class ctaid : Libptx.Expressions.Specials.ctaid, special
    {
        public static implicit operator v4_u16(ctaid ctaid) { return new v4_u16(ctaid); }
        public static implicit operator v4_s16(ctaid ctaid) { return new v4_s16(ctaid); }
        public static implicit operator v4_b16(ctaid ctaid) { return new v4_b16(ctaid); }
        public static implicit operator v4_u32(ctaid ctaid) { return new v4_u32(ctaid); }
        public static implicit operator v4_s32(ctaid ctaid) { return new v4_s32(ctaid); }
        public static implicit operator v4_b32(ctaid ctaid) { return new v4_b32(ctaid); }
    }
}
