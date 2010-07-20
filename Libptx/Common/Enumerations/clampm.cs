using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Enumerations
{
    public enum clampm
    {
        [Affix("trap")] trap = 1,
        [Affix("clamp")] clamp,
        [Affix("zero")] zero,
    }
}
