using System;
using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Types
{
    [Flags]
    public enum TypeMod
    {
        Scalar = 0,
        [Affix("v1")] V1 = 1,
        [Affix("v2")] V2 = 2,
        [Affix("v4")] V4 = 4,
        [Affix("[]")] Array = 8,
    }
}