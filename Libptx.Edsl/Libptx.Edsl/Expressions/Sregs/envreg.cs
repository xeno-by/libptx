using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Sregs
{
    public class envreg : Libptx.Expressions.Sregs.envreg, sreg
    {
        public static implicit operator u32(envreg envreg) { return new u32(envreg); }
        public static implicit operator s32(envreg envreg) { return new s32(envreg); }
        public static implicit operator f32(envreg envreg) { return new f32(envreg); }
        public static implicit operator b32(envreg envreg) { return new b32(envreg); }
    }
}
