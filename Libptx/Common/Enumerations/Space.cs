using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Enumerations
{
    public enum Space
    {
        [Affix(".reg")] Register = 0,
        [Affix(".sreg")] Special,
        [Affix(".local")] Local,
        [Affix(".shared")] Shared,
        [Affix(".global")] Global,
        [Affix(".param")] Param,
        [Affix(".const")] Const,
        [Affix(".const[0]")] Const0 = Const,
        [Affix(".const[1]")] Const1,
        [Affix(".const[2]")] Const2,
        [Affix(".const[3]")] Const3,
        [Affix(".const[4]")] Const4,
        [Affix(".const[5]")] Const5,
        [Affix(".const[6]")] Const6,
        [Affix(".const[7]")] Const7,
        [Affix(".const[8]")] Const8,
        [Affix(".const[9]")] Const9,
        [Affix(".const[10]")] Const10,
    }
}