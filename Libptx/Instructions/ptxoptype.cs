namespace Libptx.Instructions
{
    public enum ptxoptype
    {
        abs = 1,
        add,
        and,
        atom,
        bar_arrive,
        bar_red,
        bar_sync,
        bfe,
        bfi,
        bfind,
        brev,
        brkpt,
        clz,
        cnot,
        copysign,
        cos,
        cvt,
        cvta,
        div,
        ex2,
        fma,
        isspacep,
        ld,
        lg2,
        mad,
        max,
        membar,
        min,
        mov,
        mul,
        neg,
        not,
        or,
        pmevent,
        popc,
        prefetch,
        prmt,
        rcp,
        red,
        rem,
        rsqrt,
        sad,
        selp,
        set,
        shl,
        shr,
        sin,
        slct,
        sqrt,
        st,
        sub,
        suld_b,
        suld_p,
        suq,
        sured_b,
        sured_p,
        sust_b,
        sust_p,
        testp,
        tex,
        trap,
        txq,
        vabsdiff,
        vadd,
        vmad,
        vmax,
        vmin,
        vote_ballot,
        vote_pred,
        vset,
        vshl,
        vshr,
        vsub,
        xor,
    }
}

namespace Libptx.Instructions
{
    public abstract partial class ptxop
    {
        public abstract ptxoptype discr { get; }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class abs
    {
        public override ptxoptype discr { get { return ptxoptype.abs; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class add
    {
        public override ptxoptype discr { get { return ptxoptype.add; } }
    }
}

namespace Libptx.Instructions.LogicAndShift
{
    public partial class and
    {
        public override ptxoptype discr { get { return ptxoptype.and; } }
    }
}

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    public partial class atom
    {
        public override ptxoptype discr { get { return ptxoptype.atom; } }
    }
}

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    public partial class bar_arrive
    {
        public override ptxoptype discr { get { return ptxoptype.bar_arrive; } }
    }
}

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    public partial class bar_red
    {
        public override ptxoptype discr { get { return ptxoptype.bar_red; } }
    }
}

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    public partial class bar_sync
    {
        public override ptxoptype discr { get { return ptxoptype.bar_sync; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class bfe
    {
        public override ptxoptype discr { get { return ptxoptype.bfe; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class bfi
    {
        public override ptxoptype discr { get { return ptxoptype.bfi; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class bfind
    {
        public override ptxoptype discr { get { return ptxoptype.bfind; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class brev
    {
        public override ptxoptype discr { get { return ptxoptype.brev; } }
    }
}

namespace Libptx.Instructions.Miscellaneous
{
    public partial class brkpt
    {
        public override ptxoptype discr { get { return ptxoptype.brkpt; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class clz
    {
        public override ptxoptype discr { get { return ptxoptype.clz; } }
    }
}

namespace Libptx.Instructions.LogicAndShift
{
    public partial class cnot
    {
        public override ptxoptype discr { get { return ptxoptype.cnot; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class copysign
    {
        public override ptxoptype discr { get { return ptxoptype.copysign; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class cos
    {
        public override ptxoptype discr { get { return ptxoptype.cos; } }
    }
}

namespace Libptx.Instructions.MovementAndConversion
{
    public partial class cvt
    {
        public override ptxoptype discr { get { return ptxoptype.cvt; } }
    }
}

namespace Libptx.Instructions.MovementAndConversion
{
    public partial class cvta
    {
        public override ptxoptype discr { get { return ptxoptype.cvta; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class div
    {
        public override ptxoptype discr { get { return ptxoptype.div; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class ex2
    {
        public override ptxoptype discr { get { return ptxoptype.ex2; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class fma
    {
        public override ptxoptype discr { get { return ptxoptype.fma; } }
    }
}

namespace Libptx.Instructions.MovementAndConversion
{
    public partial class isspacep
    {
        public override ptxoptype discr { get { return ptxoptype.isspacep; } }
    }
}

namespace Libptx.Instructions.MovementAndConversion
{
    public partial class ld
    {
        public override ptxoptype discr { get { return ptxoptype.ld; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class lg2
    {
        public override ptxoptype discr { get { return ptxoptype.lg2; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class mad
    {
        public override ptxoptype discr { get { return ptxoptype.mad; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class max
    {
        public override ptxoptype discr { get { return ptxoptype.max; } }
    }
}

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    public partial class membar
    {
        public override ptxoptype discr { get { return ptxoptype.membar; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class min
    {
        public override ptxoptype discr { get { return ptxoptype.min; } }
    }
}

namespace Libptx.Instructions.MovementAndConversion
{
    public partial class mov
    {
        public override ptxoptype discr { get { return ptxoptype.mov; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class mul
    {
        public override ptxoptype discr { get { return ptxoptype.mul; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class neg
    {
        public override ptxoptype discr { get { return ptxoptype.neg; } }
    }
}

namespace Libptx.Instructions.LogicAndShift
{
    public partial class not
    {
        public override ptxoptype discr { get { return ptxoptype.not; } }
    }
}

namespace Libptx.Instructions.LogicAndShift
{
    public partial class or
    {
        public override ptxoptype discr { get { return ptxoptype.or; } }
    }
}

namespace Libptx.Instructions.Miscellaneous
{
    public partial class pmevent
    {
        public override ptxoptype discr { get { return ptxoptype.pmevent; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class popc
    {
        public override ptxoptype discr { get { return ptxoptype.popc; } }
    }
}

namespace Libptx.Instructions.MovementAndConversion
{
    public partial class prefetch
    {
        public override ptxoptype discr { get { return ptxoptype.prefetch; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class prmt
    {
        public override ptxoptype discr { get { return ptxoptype.prmt; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class rcp
    {
        public override ptxoptype discr { get { return ptxoptype.rcp; } }
    }
}

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    public partial class red
    {
        public override ptxoptype discr { get { return ptxoptype.red; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class rem
    {
        public override ptxoptype discr { get { return ptxoptype.rem; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class rsqrt
    {
        public override ptxoptype discr { get { return ptxoptype.rsqrt; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class sad
    {
        public override ptxoptype discr { get { return ptxoptype.sad; } }
    }
}

namespace Libptx.Instructions.ComparisonAndSelection
{
    public partial class selp
    {
        public override ptxoptype discr { get { return ptxoptype.selp; } }
    }
}

namespace Libptx.Instructions.ComparisonAndSelection
{
    public partial class set
    {
        public override ptxoptype discr { get { return ptxoptype.set; } }
    }
}

namespace Libptx.Instructions.LogicAndShift
{
    public partial class shl
    {
        public override ptxoptype discr { get { return ptxoptype.shl; } }
    }
}

namespace Libptx.Instructions.LogicAndShift
{
    public partial class shr
    {
        public override ptxoptype discr { get { return ptxoptype.shr; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class sin
    {
        public override ptxoptype discr { get { return ptxoptype.sin; } }
    }
}

namespace Libptx.Instructions.ComparisonAndSelection
{
    public partial class slct
    {
        public override ptxoptype discr { get { return ptxoptype.slct; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class sqrt
    {
        public override ptxoptype discr { get { return ptxoptype.sqrt; } }
    }
}

namespace Libptx.Instructions.MovementAndConversion
{
    public partial class st
    {
        public override ptxoptype discr { get { return ptxoptype.st; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class sub
    {
        public override ptxoptype discr { get { return ptxoptype.sub; } }
    }
}

namespace Libptx.Instructions.TextureAndSurface
{
    public partial class suld_b
    {
        public override ptxoptype discr { get { return ptxoptype.suld_b; } }
    }
}

namespace Libptx.Instructions.TextureAndSurface
{
    public partial class suld_p
    {
        public override ptxoptype discr { get { return ptxoptype.suld_p; } }
    }
}

namespace Libptx.Instructions.TextureAndSurface
{
    public partial class suq
    {
        public override ptxoptype discr { get { return ptxoptype.suq; } }
    }
}

namespace Libptx.Instructions.TextureAndSurface
{
    public partial class sured_b
    {
        public override ptxoptype discr { get { return ptxoptype.sured_b; } }
    }
}

namespace Libptx.Instructions.TextureAndSurface
{
    public partial class sured_p
    {
        public override ptxoptype discr { get { return ptxoptype.sured_p; } }
    }
}

namespace Libptx.Instructions.TextureAndSurface
{
    public partial class sust_b
    {
        public override ptxoptype discr { get { return ptxoptype.sust_b; } }
    }
}

namespace Libptx.Instructions.TextureAndSurface
{
    public partial class sust_p
    {
        public override ptxoptype discr { get { return ptxoptype.sust_p; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class testp
    {
        public override ptxoptype discr { get { return ptxoptype.testp; } }
    }
}

namespace Libptx.Instructions.TextureAndSurface
{
    public partial class tex
    {
        public override ptxoptype discr { get { return ptxoptype.tex; } }
    }
}

namespace Libptx.Instructions.Miscellaneous
{
    public partial class trap
    {
        public override ptxoptype discr { get { return ptxoptype.trap; } }
    }
}

namespace Libptx.Instructions.TextureAndSurface
{
    public partial class txq
    {
        public override ptxoptype discr { get { return ptxoptype.txq; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vabsdiff
    {
        public override ptxoptype discr { get { return ptxoptype.vabsdiff; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vadd
    {
        public override ptxoptype discr { get { return ptxoptype.vadd; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vmad
    {
        public override ptxoptype discr { get { return ptxoptype.vmad; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vmax
    {
        public override ptxoptype discr { get { return ptxoptype.vmax; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vmin
    {
        public override ptxoptype discr { get { return ptxoptype.vmin; } }
    }
}

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    public partial class vote_ballot
    {
        public override ptxoptype discr { get { return ptxoptype.vote_ballot; } }
    }
}

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    public partial class vote_pred
    {
        public override ptxoptype discr { get { return ptxoptype.vote_pred; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vset
    {
        public override ptxoptype discr { get { return ptxoptype.vset; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vshl
    {
        public override ptxoptype discr { get { return ptxoptype.vshl; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vshr
    {
        public override ptxoptype discr { get { return ptxoptype.vshr; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vsub
    {
        public override ptxoptype discr { get { return ptxoptype.vsub; } }
    }
}

namespace Libptx.Instructions.LogicAndShift
{
    public partial class xor
    {
        public override ptxoptype discr { get { return ptxoptype.xor; } }
    }
}
