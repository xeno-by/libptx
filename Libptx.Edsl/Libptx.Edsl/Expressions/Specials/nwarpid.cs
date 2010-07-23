using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Specials
{
    public class nwarpid : Libptx.Expressions.Specials.nwarpid, special
    {
        public static implicit operator u32(nwarpid nwarpid) { return new u32(nwarpid); }
        public static implicit operator s32(nwarpid nwarpid) { return new s32(nwarpid); }
        public static implicit operator b32(nwarpid nwarpid) { return new b32(nwarpid); }
    }
}
