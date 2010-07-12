using Libptx.Common.Infrastructure;

namespace Libptx.Common
{
    public enum TypeMod
    {
        [Signature(null)] Scalar = 0,
        [Signature(".v1")] V1,
        [Signature(".v2")] V2,
        [Signature(".v4")] V4,
        [Signature(null)] Array,
    }
}