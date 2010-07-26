using System;
using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Spaces
{
    [Flags]
    public enum space
    {
        [Affix("reg")] reg = 1,
        [Affix("sreg")] sreg = 2,
        [Affix("local")] local = 4,
        [Affix("shared")] shared = 8,
        [Affix("global")] global = 16,
        [Affix("param")] param = 32,
        [Affix("const")] @const = 64,
        [Affix("const[0]")] const0 = 0 | @const,
        [Affix("const[1]")] const1 = 1 | @const,
        [Affix("const[2]")] const2 = 2 | @const,
        [Affix("const[3]")] const3 = 3 | @const,
        [Affix("const[4]")] const4 = 4 | @const,
        [Affix("const[5]")] const5 = 5 | @const,
        [Affix("const[6]")] const6 = 6 | @const,
        [Affix("const[7]")] const7 = 7 | @const,
        [Affix("const[8]")] const8 = 8 | @const,
        [Affix("const[9]")] const9 = 9 | @const,
        [Affix("const[10]")] const10 = 10 | @const,
    }

    [DebuggerNonUserCode]
    public static class spaceExtensions
    {
        public static bool is_const(this space space)
        {
            return (space & space.@const) == space.@const;
        }

        public static int const_bank(this space space)
        {
            return space.is_const() ? (space - space.@const) : -1;
        }
    }
}