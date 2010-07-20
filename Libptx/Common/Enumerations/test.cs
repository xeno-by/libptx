using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Enumerations
{
    public enum test
    {
        [Affix("finite")] finite = 1,
        [Affix("infinite")] infinite,
        [Affix("number")] number,
        [Affix("notanumber")] notanumber,
        [Affix("normal")] normal,
        [Affix("subnormal")] subnormal,
    }
}
