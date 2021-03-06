using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Sregs
{
    public class pm : Libptx.Expressions.Sregs.pm, sreg
    {
        public static implicit operator u32(pm pm) { return new u32(pm); }
        public static implicit operator s32(pm pm) { return new s32(pm); }
        public static implicit operator b32(pm pm) { return new b32(pm); }
    }
}
