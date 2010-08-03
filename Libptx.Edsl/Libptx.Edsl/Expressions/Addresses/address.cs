using Libptx.Expressions.Addresses;

namespace Libptx.Edsl.Common.Types.Other
{
    public partial class bmk
    {
        public static implicit operator bmk(Label label)
        {
            return new bmk(label);
        }
    }
}
