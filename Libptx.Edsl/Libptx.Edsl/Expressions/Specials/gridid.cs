using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Specials
{
    public class gridid : Libptx.Expressions.Specials.gridid, special
    {
        public static implicit operator u32(gridid gridid) { return new u32(gridid); }
        public static implicit operator s32(gridid gridid) { return new s32(gridid); }
        public static implicit operator b32(gridid gridid) { return new b32(gridid); }
    }
}
