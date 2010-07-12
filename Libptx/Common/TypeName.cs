using Libptx.Common.Infrastructure;

namespace Libptx.Common
{
    public enum TypeName
    {
        [Signature(".u8")] U8 = 1,
        [Signature(".s8")] S8,
        [Signature(".u16")] U16,
        [Signature(".s16")] S16,
        [Signature(".u24")] U24,
        [Signature(".s24")] S24,
        [Signature(".u32")] U32,
        [Signature(".s32")] S32,
        [Signature(".u64")] U64,
        [Signature(".s64")] S64,
        [Signature(".f16")] F16,
        [Signature(".f32")] F32,
        [Signature(".f64")] F64,
        [Signature(".b8")] B8,
        [Signature(".b16")] B16,
        [Signature(".b32")] B32,
        [Signature(".b64")] B64,
        [Signature(".pred")] Pred,
        [Signature(".texref")] Tex,
        [Signature(".samplerref")] Sampler,
        [Signature(".surfref")] Surf,
    }
}