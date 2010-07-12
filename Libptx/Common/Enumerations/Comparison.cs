using Libptx.Common.Infrastructure;

namespace Libptx.Common.Enumerations
{
    public enum Comparison
    {
        [Signature("eq")] Equal = 1,
        [Signature("ne")] NotEqual,
        [Signature("lt")] LessThan,
        [Signature("le")] LessThanOrEqual,
        [Signature("gt")] GreaterThan,
        [Signature("ge")] GreaterThanOrEqual,
        [Signature("lo")] Lower,
        [Signature("ls")] LowerOrSame,
        [Signature("hi")] Higher,
        [Signature("hs")] HigherOrSame,
        [Signature("equ")] EqualUnordered,
        [Signature("neu")] NotEqualUnordered,
        [Signature("ltu")] LessThanUnordered,
        [Signature("leu")] LessThanOrEqualUnordered,
        [Signature("gtu")] GreaterThanUnordered,
        [Signature("geu")] GreaterThanOrEqualUnordered,
        [Signature("num")] BothNumbers,
        [Signature("nan")] AnyNan,
    }
}