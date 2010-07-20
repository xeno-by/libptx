using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Enumerations
{
    public enum squery
    {
        [Affix("width")] width = 1,
        [Affix("height")] height,
        [Affix("depth")] depth,
        [Affix("channel_datatype")] channel_datatype,
        [Affix("channel_order")] channel_order,
    }
}
