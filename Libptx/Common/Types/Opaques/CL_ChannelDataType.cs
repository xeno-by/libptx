using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Types.Opaques
{
    public enum CL_ChannelDataType
    {
        [Attr15("CL_SNORM_INT8")] CL_SNORM_INT8 = 0x10D0,
        [Attr15("CL_SNORM_INT16")] CL_SNORM_INT16 = 0x10D1,
        [Attr15("CL_UNORM_INT8")] CL_UNORM_INT8 = 0x10D2,
        [Attr15("CL_UNORM_INT16")] CL_UNORM_INT16 = 0x10D3,
        [Attr15("CL_UNORM_SHORT_565")] CL_UNORM_SHORT_565 = 0x10D4,
        [Attr15("CL_UNORM_SHORT_555")] CL_UNORM_SHORT_555 = 0x10D5,
        [Attr15("CL_UNORM_INT_101010")] CL_UNORM_INT_101010 = 0x10D6,
        [Attr15("CL_SIGNED_INT8")] CL_SIGNED_INT8 = 0x10D7,
        [Attr15("CL_SIGNED_INT16")] CL_SIGNED_INT16 = 0x10D8,
        [Attr15("CL_SIGNED_INT32")] CL_SIGNED_INT32 = 0x10D9,
        [Attr15("CL_UNSIGNED_INT8")] CL_UNSIGNED_INT8 = 0x10DA,
        [Attr15("CL_UNSIGNED_INT16")] CL_UNSIGNED_INT16 = 0x10DB,
        [Attr15("CL_UNSIGNED_INT32")] CL_UNSIGNED_INT32 = 0x10DC,
        [Attr15("CL_HALF_FLOAT")] CL_HALF_FLOAT = 0x10DD,
        [Attr15("CL_FLOAT")] CL_FLOAT = 0x10DE,
    }
}