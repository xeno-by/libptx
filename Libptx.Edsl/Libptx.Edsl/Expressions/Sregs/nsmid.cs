using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Sregs
{
    public class nsmid : Libptx.Expressions.Sregs.nsmid, sreg
    {
        public static implicit operator u32(nsmid nsmid) { return new u32(nsmid); }
        public static implicit operator s32(nsmid nsmid) { return new s32(nsmid); }
        public static implicit operator b32(nsmid nsmid) { return new b32(nsmid); }
    }
}
