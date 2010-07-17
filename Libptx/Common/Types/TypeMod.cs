using Libptx.Common.Annotations.Quantas;

namespace Libptx.Common.Types
{
    public enum TypeMod
    {
        [Affix(null)] Scalar = 0,
        [Affix(".v1")] V1,
        [Affix(".v2")] V2,
        [Affix(".v4")] V4,
        [Affix(null)] Array,
    }
}