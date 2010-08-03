using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Sregs
{
    public class smid : Libptx.Expressions.Sregs.smid, sreg
    {
        public static implicit operator u32(smid smid) { return new u32(smid); }
        public static implicit operator s32(smid smid) { return new s32(smid); }
        public static implicit operator b32(smid smid) { return new b32(smid); }
    }
}
