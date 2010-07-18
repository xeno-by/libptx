using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Types
{
    public enum TypeMod
    {
        [Affix(null)] Scalar = 0,
        [Prefix(".v1")] V1,
        [Prefix(".v2")] V2,
        [Prefix(".v4")] V4,
        [Suffix(null)] Array = 8,
    }
}