using Libptx.Common.Annotations.Quantas;

namespace Libptx.Common.Enumerations
{
    public enum Comparison
    {
        [Affix("eq")] Equal = 1,
        [Affix("ne")] NotEqual,
        [Affix("lt")] LessThan,
        [Affix("le")] LessThanOrEqual,
        [Affix("gt")] GreaterThan,
        [Affix("ge")] GreaterThanOrEqual,
        [Affix("lo")] Lower,
        [Affix("ls")] LowerOrSame,
        [Affix("hi")] Higher,
        [Affix("hs")] HigherOrSame,
        [Affix("equ")] EqualUnordered,
        [Affix("neu")] NotEqualUnordered,
        [Affix("ltu")] LessThanUnordered,
        [Affix("leu")] LessThanOrEqualUnordered,
        [Affix("gtu")] GreaterThanUnordered,
        [Affix("geu")] GreaterThanOrEqualUnordered,
        [Affix("num")] BothNumbers,
        [Affix("nan")] AnyNan,
    }
}