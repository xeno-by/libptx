using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Specials
{
    public class lanemask : Libptx.Expressions.Specials.lanemask, special
    {
        public static implicit operator u32(lanemask lanemask) { return new u32(lanemask); }
        public static implicit operator s32(lanemask lanemask) { return new s32(lanemask); }
        public static implicit operator b32(lanemask lanemask) { return new b32(lanemask); }
    }
}
