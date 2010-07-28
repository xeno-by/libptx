using System;
using Libptx.Common.Annotations.Quanta;

namespace Libptx.Expressions
{
    [Flags]
    public enum Mod
    {
        [Affix("!")] Not = 1,
        [Affix("|")] Couple = 2,
        [Affix("-")] Neg = 4,
        [Affix("b0")] B0 = 8,
        [Affix("b1")] B1 = 16,
        [Affix("b2")] B2 = 32,
        [Affix("b3")] B3 = 64,
        [Affix("h0")] H0 = 128,
        [Affix("h1")] H1 = 256,
        [Affix("x")] X = 512,
        [Affix("r")] R = X,
        [Affix("y")] Y = 1024,
        [Affix("g")] G = Y,
        [Affix("z")] Z = 2048,
        [Affix("b")] B = Z,
        [Affix("w")] W = 4096,
        [Affix("a")] A = W,
    }
}