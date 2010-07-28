namespace Libptx.Expressions.Sregs
{
    public enum sregtype
    {
        clock32 = 1,
        clock64,
        ctaid,
        envreg,
        gridid,
        laneid,
        lanemask,
        nctaid,
        nsmid,
        ntid,
        nwarpid,
        pm,
        smid,
        tid,
        warpid,
    }
}

namespace Libptx.Expressions.Sregs
{
    public abstract partial class Sreg
    {
        public abstract sregtype discr { get; }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class clock32
    {
        public override sregtype discr { get { return sregtype.clock32; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class clock64
    {
        public override sregtype discr { get { return sregtype.clock64; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class ctaid
    {
        public override sregtype discr { get { return sregtype.ctaid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class envreg
    {
        public override sregtype discr { get { return sregtype.envreg; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class gridid
    {
        public override sregtype discr { get { return sregtype.gridid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class laneid
    {
        public override sregtype discr { get { return sregtype.laneid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class lanemask
    {
        public override sregtype discr { get { return sregtype.lanemask; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class nctaid
    {
        public override sregtype discr { get { return sregtype.nctaid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class nsmid
    {
        public override sregtype discr { get { return sregtype.nsmid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class ntid
    {
        public override sregtype discr { get { return sregtype.ntid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class nwarpid
    {
        public override sregtype discr { get { return sregtype.nwarpid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class pm
    {
        public override sregtype discr { get { return sregtype.pm; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class smid
    {
        public override sregtype discr { get { return sregtype.smid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class tid
    {
        public override sregtype discr { get { return sregtype.tid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class warpid
    {
        public override sregtype discr { get { return sregtype.warpid; } }
    }
}
