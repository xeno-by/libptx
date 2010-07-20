using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Enumerations
{
    public enum cmp
    {
        [Affix("eq")] eq = 1,
        [Affix("ne")] ne,
        [Affix("lt")] lt,
        [Affix("le")] le,
        [Affix("gt")] gt,
        [Affix("ge")] ge,
        [Affix("lo")] lo,
        [Affix("ls")] ls,
        [Affix("hi")] hi,
        [Affix("hs")] hs,
        [Affix("equ")] equ,
        [Affix("neu")] neu,
        [Affix("ltu")] ltu,
        [Affix("leu")] leu,
        [Affix("gtu")] gtu,
        [Affix("geu")] geu,
        [Affix("num")] num,
        [Affix("nan")] nan,
    }
}