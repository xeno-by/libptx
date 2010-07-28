using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Types.Opaques
{
    public enum CL_ChannelDataType
    {
        [Affix15("CL_SNORM_INT8")] CL_SNORM_INT8 = 0x10D0,
        [Affix15("CL_SNORM_INT16")] CL_SNORM_INT16 = 0x10D1,
        [Affix15("CL_UNORM_INT8")] CL_UNORM_INT8 = 0x10D2,
        [Affix15("CL_UNORM_INT16")] CL_UNORM_INT16 = 0x10D3,
        [Affix15("CL_UNORM_SHORT_565")] CL_UNORM_SHORT_565 = 0x10D4,
        [Affix15("CL_UNORM_SHORT_555")] CL_UNORM_SHORT_555 = 0x10D5,
        [Affix15("CL_UNORM_INT_101010")] CL_UNORM_INT_101010 = 0x10D6,
        [Affix15("CL_SIGNED_INT8")] CL_SIGNED_INT8 = 0x10D7,
        [Affix15("CL_SIGNED_INT16")] CL_SIGNED_INT16 = 0x10D8,
        [Affix15("CL_SIGNED_INT32")] CL_SIGNED_INT32 = 0x10D9,
        [Affix15("CL_UNSIGNED_INT8")] CL_UNSIGNED_INT8 = 0x10DA,
        [Affix15("CL_UNSIGNED_INT16")] CL_UNSIGNED_INT16 = 0x10DB,
        [Affix15("CL_UNSIGNED_INT32")] CL_UNSIGNED_INT32 = 0x10DC,
        [Affix15("CL_HALF_FLOAT")] CL_HALF_FLOAT = 0x10DD,
        [Affix15("CL_FLOAT")] CL_FLOAT = 0x10DE,
    }
}