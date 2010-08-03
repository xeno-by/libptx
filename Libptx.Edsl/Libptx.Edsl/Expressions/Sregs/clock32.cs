using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Sregs
{
    public class clock32 : Libptx.Expressions.Sregs.clock32, sreg
    {
        public static implicit operator u32(clock32 clock32) { return new u32(clock32); }
        public static implicit operator s32(clock32 clock32) { return new s32(clock32); }
        public static implicit operator b32(clock32 clock32) { return new b32(clock32); }
    }
}
