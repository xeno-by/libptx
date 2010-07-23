using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Types
{
    public enum TypeName
    {
        [Affix("u8")] U8 = 1,
        [Affix("s8")] S8,
        [Affix("u16")] U16,
        [Affix("s16")] S16,
        [Affix("u32")] U32,
        [Affix("s32")] S32,
        [Affix("u64")] U64,
        [Affix("s64")] S64,
        [Affix("f16")] F16,
        [Affix("f32")] F32,
        [Affix("f64", SoftwareIsa.PTX_14, HardwareIsa.SM_13)] F64,
        [Affix("b8")] B8,
        [Affix("b16")] B16,
        [Affix("b32")] B32,
        [Affix("b64")] B64,
        [Affix("pred")] Pred,
        [Affix("texref", SoftwareIsa.PTX_15)] Texref,
        [Affix("samplerref", SoftwareIsa.PTX_15)] Samplerref,
        [Affix("surfref", SoftwareIsa.PTX_15)] Surfref,
    }
}