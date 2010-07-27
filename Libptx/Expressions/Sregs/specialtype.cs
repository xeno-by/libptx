using Libptx.Expressions.Sregs;

namespace Libptx.Expressions.Sregs
{
    public enum specialtype
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
        public abstract specialtype discr { get; }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class clock32
    {
        public override specialtype discr { get { return specialtype.clock32; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class clock64
    {
        public override specialtype discr { get { return specialtype.clock64; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class ctaid
    {
        public override specialtype discr { get { return specialtype.ctaid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class envreg
    {
        public override specialtype discr { get { return specialtype.envreg; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class gridid
    {
        public override specialtype discr { get { return specialtype.gridid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class laneid
    {
        public override specialtype discr { get { return specialtype.laneid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class lanemask
    {
        public override specialtype discr { get { return specialtype.lanemask; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class nctaid
    {
        public override specialtype discr { get { return specialtype.nctaid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class nsmid
    {
        public override specialtype discr { get { return specialtype.nsmid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class ntid
    {
        public override specialtype discr { get { return specialtype.ntid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class nwarpid
    {
        public override specialtype discr { get { return specialtype.nwarpid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class pm
    {
        public override specialtype discr { get { return specialtype.pm; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class smid
    {
        public override specialtype discr { get { return specialtype.smid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class tid
    {
        public override specialtype discr { get { return specialtype.tid; } }
    }
}

namespace Libptx.Expressions.Sregs
{
    public partial class warpid
    {
        public override specialtype discr { get { return specialtype.warpid; } }
    }
}
