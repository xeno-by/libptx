using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Enumerations
{
    public enum tquery
    {
        [Affix("width")] width = 1,
        [Affix("height")] height,
        [Affix("depth")] depth,
        [Affix("channel_datatype")] channel_datatype,
        [Affix("channel_order")] channel_order,
        [Affix("normalized_coords")] normalized_coords,
        [Affix("filter_mode")] filter_mode,
        [Affix("addr_mode_0")] addr_mode_0,
        [Affix("addr_mode_1")] addr_mode_1,
        [Affix("addr_mode_2")] addr_mode_2,
    }
}
