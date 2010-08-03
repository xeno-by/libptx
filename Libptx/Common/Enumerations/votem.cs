using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Enumerations
{
    public enum votem
    {
        [Affix("all")] ballot = 1,
        [Affix("all")] all,
        [Affix("any")] any,
        [Affix("uni")] uni,
    }
}
