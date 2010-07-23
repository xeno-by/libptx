using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Specials
{
    public class nsmid : Libptx.Expressions.Specials.nsmid, special
    {
        public static implicit operator u32(nsmid nsmid) { return new u32(nsmid); }
        public static implicit operator s32(nsmid nsmid) { return new s32(nsmid); }
        public static implicit operator b32(nsmid nsmid) { return new b32(nsmid); }
    }
}
