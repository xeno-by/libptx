namespace Libptx.Instructions
{
    public enum ptxoptype
    {
        vshl,
        bar_arrive,
        abs,
        fma,
        sured,
        membar,
        not,
        testp,
        prmt,
        lg2,
        rem,
        slct,
        vmax,
        st,
        isspacep,
        sin,
        cos,
        vadd,
        vmin,
        copysign,
        vsub,
        vset,
        tex,
        sust,
        bar_red,
        set,
        selp,
        sad,
        bfi,
        vmad,
        red,
        vote_pred,
        popc,
        mul,
        shr,
        prefetch,
        ex2,
        bar_sync,
        cnot,
        trap,
        sqrt,
        max,
        vote_ballot,
        xor,
        cvta,
        min,
        bfind,
        bfe,
        add,
        vshr,
        sub,
        rsqrt,
        div,
        clz,
        and,
        pmevent,
        neg,
        mad,
        suq,
        or,
        ld,
        atom,
        shl,
        mov,
        brkpt,
        rcp,
        vabsdiff,
        txq,
        suld,
        brev,
        cvt,
    }
}

namespace Libptx.Instructions
{
    public abstract partial class ptxop
    {
        public abstract ptxoptype optype { get; }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vshl
    {
        public override ptxoptype optype { get { return ptxoptype.vshl; } }
    }
}

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    public partial class bar_arrive
    {
        public override ptxoptype optype { get { return ptxoptype.bar_arrive; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class abs
    {
        public override ptxoptype optype { get { return ptxoptype.abs; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class fma
    {
        public override ptxoptype optype { get { return ptxoptype.fma; } }
    }
}

namespace Libptx.Instructions.TextureAndSurface
{
    public partial class sured
    {
        public override ptxoptype optype { get { return ptxoptype.sured; } }
    }
}

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    public partial class membar
    {
        public override ptxoptype optype { get { return ptxoptype.membar; } }
    }
}

namespace Libptx.Instructions.LogicAndShift
{
    public partial class not
    {
        public override ptxoptype optype { get { return ptxoptype.not; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class testp
    {
        public override ptxoptype optype { get { return ptxoptype.testp; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class prmt
    {
        public override ptxoptype optype { get { return ptxoptype.prmt; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class lg2
    {
        public override ptxoptype optype { get { return ptxoptype.lg2; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class rem
    {
        public override ptxoptype optype { get { return ptxoptype.rem; } }
    }
}

namespace Libptx.Instructions.ComparisonAndSelection
{
    public partial class slct
    {
        public override ptxoptype optype { get { return ptxoptype.slct; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vmax
    {
        public override ptxoptype optype { get { return ptxoptype.vmax; } }
    }
}

namespace Libptx.Instructions.MovementAndConversion
{
    public partial class st
    {
        public override ptxoptype optype { get { return ptxoptype.st; } }
    }
}

namespace Libptx.Instructions.MovementAndConversion
{
    public partial class isspacep
    {
        public override ptxoptype optype { get { return ptxoptype.isspacep; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class sin
    {
        public override ptxoptype optype { get { return ptxoptype.sin; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class cos
    {
        public override ptxoptype optype { get { return ptxoptype.cos; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vadd
    {
        public override ptxoptype optype { get { return ptxoptype.vadd; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vmin
    {
        public override ptxoptype optype { get { return ptxoptype.vmin; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class copysign
    {
        public override ptxoptype optype { get { return ptxoptype.copysign; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vsub
    {
        public override ptxoptype optype { get { return ptxoptype.vsub; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vset
    {
        public override ptxoptype optype { get { return ptxoptype.vset; } }
    }
}

namespace Libptx.Instructions.TextureAndSurface
{
    public partial class tex
    {
        public override ptxoptype optype { get { return ptxoptype.tex; } }
    }
}

namespace Libptx.Instructions.TextureAndSurface
{
    public partial class sust
    {
        public override ptxoptype optype { get { return ptxoptype.sust; } }
    }
}

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    public partial class bar_red
    {
        public override ptxoptype optype { get { return ptxoptype.bar_red; } }
    }
}

namespace Libptx.Instructions.ComparisonAndSelection
{
    public partial class set
    {
        public override ptxoptype optype { get { return ptxoptype.set; } }
    }
}

namespace Libptx.Instructions.ComparisonAndSelection
{
    public partial class selp
    {
        public override ptxoptype optype { get { return ptxoptype.selp; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class sad
    {
        public override ptxoptype optype { get { return ptxoptype.sad; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class bfi
    {
        public override ptxoptype optype { get { return ptxoptype.bfi; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vmad
    {
        public override ptxoptype optype { get { return ptxoptype.vmad; } }
    }
}

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    public partial class red
    {
        public override ptxoptype optype { get { return ptxoptype.red; } }
    }
}

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    public partial class vote_pred
    {
        public override ptxoptype optype { get { return ptxoptype.vote_pred; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class popc
    {
        public override ptxoptype optype { get { return ptxoptype.popc; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class mul
    {
        public override ptxoptype optype { get { return ptxoptype.mul; } }
    }
}

namespace Libptx.Instructions.LogicAndShift
{
    public partial class shr
    {
        public override ptxoptype optype { get { return ptxoptype.shr; } }
    }
}

namespace Libptx.Instructions.MovementAndConversion
{
    public partial class prefetch
    {
        public override ptxoptype optype { get { return ptxoptype.prefetch; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class ex2
    {
        public override ptxoptype optype { get { return ptxoptype.ex2; } }
    }
}

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    public partial class bar_sync
    {
        public override ptxoptype optype { get { return ptxoptype.bar_sync; } }
    }
}

namespace Libptx.Instructions.LogicAndShift
{
    public partial class cnot
    {
        public override ptxoptype optype { get { return ptxoptype.cnot; } }
    }
}

namespace Libptx.Instructions.Miscellaneous
{
    public partial class trap
    {
        public override ptxoptype optype { get { return ptxoptype.trap; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class sqrt
    {
        public override ptxoptype optype { get { return ptxoptype.sqrt; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class max
    {
        public override ptxoptype optype { get { return ptxoptype.max; } }
    }
}

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    public partial class vote_ballot
    {
        public override ptxoptype optype { get { return ptxoptype.vote_ballot; } }
    }
}

namespace Libptx.Instructions.LogicAndShift
{
    public partial class xor
    {
        public override ptxoptype optype { get { return ptxoptype.xor; } }
    }
}

namespace Libptx.Instructions.MovementAndConversion
{
    public partial class cvta
    {
        public override ptxoptype optype { get { return ptxoptype.cvta; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class min
    {
        public override ptxoptype optype { get { return ptxoptype.min; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class bfind
    {
        public override ptxoptype optype { get { return ptxoptype.bfind; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class bfe
    {
        public override ptxoptype optype { get { return ptxoptype.bfe; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class add
    {
        public override ptxoptype optype { get { return ptxoptype.add; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vshr
    {
        public override ptxoptype optype { get { return ptxoptype.vshr; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class sub
    {
        public override ptxoptype optype { get { return ptxoptype.sub; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class rsqrt
    {
        public override ptxoptype optype { get { return ptxoptype.rsqrt; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class div
    {
        public override ptxoptype optype { get { return ptxoptype.div; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class clz
    {
        public override ptxoptype optype { get { return ptxoptype.clz; } }
    }
}

namespace Libptx.Instructions.LogicAndShift
{
    public partial class and
    {
        public override ptxoptype optype { get { return ptxoptype.and; } }
    }
}

namespace Libptx.Instructions.Miscellaneous
{
    public partial class pmevent
    {
        public override ptxoptype optype { get { return ptxoptype.pmevent; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class neg
    {
        public override ptxoptype optype { get { return ptxoptype.neg; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class mad
    {
        public override ptxoptype optype { get { return ptxoptype.mad; } }
    }
}

namespace Libptx.Instructions.TextureAndSurface
{
    public partial class suq
    {
        public override ptxoptype optype { get { return ptxoptype.suq; } }
    }
}

namespace Libptx.Instructions.LogicAndShift
{
    public partial class or
    {
        public override ptxoptype optype { get { return ptxoptype.or; } }
    }
}

namespace Libptx.Instructions.MovementAndConversion
{
    public partial class ld
    {
        public override ptxoptype optype { get { return ptxoptype.ld; } }
    }
}

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    public partial class atom
    {
        public override ptxoptype optype { get { return ptxoptype.atom; } }
    }
}

namespace Libptx.Instructions.LogicAndShift
{
    public partial class shl
    {
        public override ptxoptype optype { get { return ptxoptype.shl; } }
    }
}

namespace Libptx.Instructions.MovementAndConversion
{
    public partial class mov
    {
        public override ptxoptype optype { get { return ptxoptype.mov; } }
    }
}

namespace Libptx.Instructions.Miscellaneous
{
    public partial class brkpt
    {
        public override ptxoptype optype { get { return ptxoptype.brkpt; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class rcp
    {
        public override ptxoptype optype { get { return ptxoptype.rcp; } }
    }
}

namespace Libptx.Instructions.Video
{
    public partial class vabsdiff
    {
        public override ptxoptype optype { get { return ptxoptype.vabsdiff; } }
    }
}

namespace Libptx.Instructions.TextureAndSurface
{
    public partial class txq
    {
        public override ptxoptype optype { get { return ptxoptype.txq; } }
    }
}

namespace Libptx.Instructions.TextureAndSurface
{
    public partial class suld
    {
        public override ptxoptype optype { get { return ptxoptype.suld; } }
    }
}

namespace Libptx.Instructions.Arithmetic
{
    public partial class brev
    {
        public override ptxoptype optype { get { return ptxoptype.brev; } }
    }
}

namespace Libptx.Instructions.MovementAndConversion
{
    public partial class cvt
    {
        public override ptxoptype optype { get { return ptxoptype.cvt; } }
    }
}
