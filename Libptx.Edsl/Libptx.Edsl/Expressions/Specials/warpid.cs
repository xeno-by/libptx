using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Specials
{
    public class warpid : Libptx.Expressions.Specials.warpid, special
    {
        public static implicit operator u32(warpid warpid) { return new u32(warpid); }
        public static implicit operator s32(warpid warpid) { return new s32(warpid); }
        public static implicit operator b32(warpid warpid) { return new b32(warpid); }
    }
}
