using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Types.Opaques
{
    public enum CL_ChannelOrder
    {
        [Attr15("CL_R")] CL_R = 0x10B0,
        [Attr15("CL_A")] CL_A = 0x10B1,
        [Attr15("CL_RG")] CL_RG = 0x10B2,
        [Attr15("CL_RA")] CL_RA = 0x10B3,
        [Attr15("CL_RGB")] CL_RGB = 0x10B4,
        [Attr15("CL_RGBA")] CL_RGBA = 0x10B5,
        [Attr15("CL_BGRA")] CL_BGRA = 0x10B6,
        [Attr15("CL_ARGB")] CL_ARGB = 0x10B7,
        [Attr15("CL_INTENSITY")] CL_INTENSITY = 0x10B8,
        [Attr15("CL_LUMINANCE")] CL_LUMINANCE = 0x10B9,
    }
}