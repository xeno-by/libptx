using Libptx.Common.Annotations.Quanta;

namespace Libptx.Common.Enumerations
{
    public enum space
    {
        [Affix("reg")] reg = 0,
        [Affix("sreg")] sreg,
        [Affix("local")] local,
        [Affix("shared")] shared,
        [Affix("global")] global,
        [Affix("param")] param,
        [Affix("const")] @const,
        [Affix("const[0]")] const0 = @const,
        [Affix("const[1]")] const1,
        [Affix("const[2]")] const2,
        [Affix("const[3]")] const3,
        [Affix("const[4]")] const4,
        [Affix("const[5]")] const5,
        [Affix("const[6]")] const6,
        [Affix("const[7]")] const7,
        [Affix("const[8]")] const8,
        [Affix("const[9]")] const9,
        [Affix("const[10]")] const10,
    }
}