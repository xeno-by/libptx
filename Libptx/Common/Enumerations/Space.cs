using Libptx.Common.Infrastructure;

namespace Libptx.Common.Enumerations
{
    public enum Space
    {
        [Signature(".reg")] Register = 0,
        [Signature(".sreg")] Special,
        [Signature(".local")] Local,
        [Signature(".shared")] Shared,
        [Signature(".global")] Global,
        [Signature(".param")] Param,
        [Signature(".const")] Const,
        [Signature(".const[0]")] Const0 = Const,
        [Signature(".const[1]")] Const1,
        [Signature(".const[2]")] Const2,
        [Signature(".const[3]")] Const3,
        [Signature(".const[4]")] Const4,
        [Signature(".const[5]")] Const5,
        [Signature(".const[6]")] Const6,
        [Signature(".const[7]")] Const7,
        [Signature(".const[8]")] Const8,
        [Signature(".const[9]")] Const9,
        [Signature(".const[10]")] Const10,
    }
}