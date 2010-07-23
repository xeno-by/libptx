using Libptx.Edsl.Common.Types.Scalar;
using Libptx.Edsl.Common.Types.Vector;

namespace Libptx.Edsl.Expressions.Specials
{
    public class clock64 : Libptx.Expressions.Specials.clock64, special
    {
        public static implicit operator u64(clock64 clock64) { return new u64(clock64); }
        public static implicit operator s64(clock64 clock64) { return new s64(clock64); }
        public static implicit operator b64(clock64 clock64) { return new b64(clock64); }
    }
}
