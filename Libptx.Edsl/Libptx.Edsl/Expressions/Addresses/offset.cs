using Libptx.Expressions.Addresses;

namespace Libptx.Edsl.Common.Types.Other
{
    public partial class ptr
    {
        public static implicit operator ptr(Offset offset)
        {
            return new ptr(offset);
        }
    }
}
