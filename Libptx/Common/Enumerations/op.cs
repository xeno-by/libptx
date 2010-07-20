using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Enumerations
{
    public enum op
    {
        [Affix("add")] add = 1,
        [Affix("min")] min,
        [Affix("max")] max,
        [Affix("and")] and,
        [Affix("or")] or,
        [Affix("xor")] xor,
        [Affix("cas")] cas,
        [Affix("exch")] exch,
        [Affix("inc")] inc,
        [Affix("dec")] dec,
    }
}
