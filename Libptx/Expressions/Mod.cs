using System;
using Libptx.Common.Annotations.Quanta;

namespace Libptx.Expressions
{
    [Flags]
    public enum Mod
    {
        [Prefix("!")] Not = 1,
        [Infix("|")] Couple,
        [Prefix("-")] Neg,
        [Affix("b0")] B0,
        [Affix("b1")] B1,
        [Affix("b2")] B2,
        [Affix("b3")] B3,
        [Affix("h0")] H0,
        [Affix("h1")] H1,
        [Affix("x")] X,
        [Affix("r")] R = X,
        [Affix("y")] Y,
        [Affix("g")] G = Y,
        [Affix("z")] Z,
        [Affix("b")] B = Z,
        [Affix("w")] W,
        [Affix("a")] A = W,
    }
}