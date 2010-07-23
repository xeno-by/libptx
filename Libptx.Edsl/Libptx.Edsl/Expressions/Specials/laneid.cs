using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Specials
{
    public class laneid : Libptx.Expressions.Specials.laneid, special
    {
        public static implicit operator u32(laneid laneid) { return new u32(laneid); }
        public static implicit operator s32(laneid laneid) { return new s32(laneid); }
        public static implicit operator b32(laneid laneid) { return new b32(laneid); }
    }
}
