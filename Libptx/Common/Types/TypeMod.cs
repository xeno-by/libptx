using System;
using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Types
{
    [Flags]
    public enum TypeMod
    {
        [Affix(null)] Scalar = 0,
        [Prefix("v1")] V1 = 1,
        [Prefix("v2")] V2 = 2,
        [Prefix("v4")] V4 = 4,
        [Suffix(null)] Array = 8,
    }
}