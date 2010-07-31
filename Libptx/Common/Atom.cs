using System;
using System.Collections.Generic;
using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Comments;
using Libptx.Common.Contexts;
using Libptx.Common.Enumerations;
using Libptx.Common.Performance.Pragmas;
using Libptx.Common.Types;
using Libptx.Expressions;
using Libptx.Functions;
using Libptx.Reflection;
using Libptx.Statements;
using XenoGears.Strings.Writers;
using Type=Libptx.Common.Types.Type;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Common
{
    [DebuggerNonUserCode]
    public abstract class Atom : Validatable, Renderable
    {
        protected Context ctx { get { return (Context)ValidationContext.Current ?? (Context)RenderPtxContext.Current; } }
        protected DelayedWriter writer { get { return RenderPtxContext.Current == null ? null : RenderPtxContext.Current.Writer; } }
        protected IndentedWriter indented { get { return writer == null ? null : writer.InnerWriter.AssertCast<IndentedWriter>(); } }

        private IList<Comment> _comments = new List<Comment>();
        public IList<Comment> Comments
        {
            get { return _comments; }
            set { _comments = value ?? new List<Comment>(); }
        }

        private IList<Pragma> _pragmas = new List<Pragma>();
        public IList<Pragma> Pragmas
        {
            get { return _pragmas; }
            set { _pragmas = value ?? new List<Pragma>(); }
        }

        public SoftwareIsa Version { get { return (SoftwareIsa)Math.Max((int)this.EigenVersion(), (int)CustomVersion); } }
        protected virtual SoftwareIsa CustomVersion { get { return SoftwareIsa.PTX_10; } }

        public HardwareIsa Target { get { return (HardwareIsa)Math.Max((int)this.EigenTarget(), (int)CustomTarget); } }
        protected virtual HardwareIsa CustomTarget { get { return HardwareIsa.SM_10; } }

        void Validatable.Validate() { using (ctx.Push(this)) { CoreValidate(); CustomValidate(); } }
        protected virtual void CustomValidate() { /* do nothing */ }
        protected void CoreValidate()
        {
            (ctx.Version >= Version).AssertTrue();
            (ctx.Target >= Target).AssertTrue();

            Comments.ForEach(c => { c.AssertNotNull(); c.Validate(); });
            Pragmas.IsNotEmpty().AssertImplies(this is Instruction || this is Entry);
            Pragmas.ForEach(p => { p.AssertNotNull(); p.Validate(); });
        }

        protected abstract void RenderPtx();
        void Renderable.RenderPtx()
        {
            using (ctx.Push(this))
            {
                Comments.ForEach(c => c.RenderPtx());
                RenderPtx();
            }
        }

        protected abstract void RenderCubin();
        void Renderable.RenderCubin()
        {
            using (ctx.Push(this))
            {
                Comments.ForEach(c => c.RenderCubin());
                RenderCubin();
            }
        }

        public override String ToString()
        {
            return this.PeekRenderPtx();
        }

        #region Enumeration values => Static properties

        protected static Mod not { get { return Mod.Not; } }
        protected static Mod couple { get { return Mod.Couple; } }
        protected static Mod neg { get { return Mod.Neg; } }
        protected static Mod sel { get { return Mod.B0 | Mod.B1 | Mod.B2 | Mod.B3 | Mod.H0 | Mod.H1; } }
        protected static Mod member { get { return Mod.X | Mod.R | Mod.Y | Mod.G | Mod.Z | Mod.B | Mod.W | Mod.A; } }

        protected static Type u8 { get { return new Type { Name = TypeName.U8 }; } }
        protected static Type s8 { get { return new Type { Name = TypeName.S8 }; } }
        protected static Type u16 { get { return new Type { Name = TypeName.U16 }; } }
        protected static Type s16 { get { return new Type { Name = TypeName.S16 }; } }
        protected static Type u32 { get { return new Type { Name = TypeName.U32 }; } }
        protected static Type s32 { get { return new Type { Name = TypeName.S32 }; } }
        protected static Type u64 { get { return new Type { Name = TypeName.U64 }; } }
        protected static Type s64 { get { return new Type { Name = TypeName.S64 }; } }
        protected static Type f16 { get { return new Type { Name = TypeName.F16 }; } }
        protected static Type f32 { get { return new Type { Name = TypeName.F32 }; } }
        protected static Type f64 { get { return new Type { Name = TypeName.F64 }; } }
        protected static Type b8 { get { return new Type { Name = TypeName.B8 }; } }
        protected static Type b16 { get { return new Type { Name = TypeName.B16 }; } }
        protected static Type b32 { get { return new Type { Name = TypeName.B32 }; } }
        protected static Type b64 { get { return new Type { Name = TypeName.B64 }; } }
        protected static Type pred { get { return new Type { Name = TypeName.Pred }; } }
        // note. use the is_texref extension method to check whether some expression is of tex type
        // note. use the is_samplerref extension method to check whether some expression is of sampler type
        // note. use the is_surfref extension method to check whether some expression is of surf type
        // note. use the is_ptr extension method to check whether some expression is of pointer type
        // note. use the is_bmk extension method to check whether some expression is of bookmark type

        protected static barlevel cta { get { return barlevel.cta; } }
        protected static barlevel gl { get { return barlevel.gl; } }
        protected static barlevel sys { get { return barlevel.sys; } }

        protected static cachelevel L1 { get { return cachelevel.L1; } }
        protected static cachelevel L2 { get { return cachelevel.L2; } }

        protected static clampm clamp_trap { get { return clampm.trap; } }
        protected static clampm clamp_clamp { get { return clampm.clamp; } }
        protected static clampm clamp_zero { get { return clampm.zero; } }

        protected static cmp eq { get { return cmp.eq; } }
        protected static cmp ne { get { return cmp.ne; } }
        protected static cmp lt { get { return cmp.lt; } }
        protected static cmp le { get { return cmp.le; } }
        protected static cmp gt { get { return cmp.gt; } }
        protected static cmp ge { get { return cmp.ge; } }
        protected static cmp lo { get { return cmp.lo; } }
        protected static cmp ls { get { return cmp.ls; } }
        protected static cmp hi { get { return cmp.hi; } }
        protected static cmp hs { get { return cmp.hs; } }
        protected static cmp equ { get { return cmp.equ; } }
        protected static cmp neu { get { return cmp.neu; } }
        protected static cmp ltu { get { return cmp.ltu; } }
        protected static cmp leu { get { return cmp.leu; } }
        protected static cmp gtu { get { return cmp.gtu; } }
        protected static cmp geu { get { return cmp.geu; } }
        protected static cmp num { get { return cmp.num; } }
        protected static cmp nan { get { return cmp.nan; } }

        protected static cop ca { get { return cop.ca; } }
        protected static cop cg { get { return cop.cg; } }
        protected static cop cs { get { return cop.cs; } }
        protected static cop lu { get { return cop.lu; } }
        protected static cop cv { get { return cop.cv; } }
        protected static cop wb { get { return cop.wb; } }
        protected static cop wt { get { return cop.wt; } }

        protected static frnd rn { get { return frnd.rn; } }
        protected static frnd rz { get { return frnd.rz; } }
        protected static frnd rm { get { return frnd.rm; } }
        protected static frnd rp { get { return frnd.rp; } }

        protected static geom d1 { get { return geom.d1; } }
        protected static geom d2 { get { return geom.d2; } }
        protected static geom d3 { get { return geom.d3; } }

        protected static irnd rni { get { return irnd.rni; } }
        protected static irnd rzi { get { return irnd.rzi; } }
        protected static irnd rmi { get { return irnd.rmi; } }
        protected static irnd rpi { get { return irnd.rpi; } }

        protected static mulm mulm_hi { get { return mulm.hi; } }
        protected static mulm mulm_lo { get { return mulm.lo; } }
        protected static mulm wide { get { return mulm.wide; } }

        protected static op add { get { return op.add; } }
        protected static op min { get { return op.min; } }
        protected static op max { get { return op.max; } }
        protected static op and { get { return op.and; } }
        protected static op or { get { return op.or; } }
        protected static op xor { get { return op.xor; } }
        protected static op cas { get { return op.cas; } }
        protected static op exch { get { return op.exch; } }
        protected static op inc { get { return op.inc; } }
        protected static op dec { get { return op.dec; } }
        protected static op popc { get { return op.popc; } }

        protected static prmtm f4e { get { return prmtm.f4e; } }
        protected static prmtm b4e { get { return prmtm.b4e; } }
        protected static prmtm rc8 { get { return prmtm.rc8; } }
        protected static prmtm ec1 { get { return prmtm.ec1; } }
        protected static prmtm ecr { get { return prmtm.ecr; } }
        protected static prmtm rc16 { get { return prmtm.rc16; } }

        protected static redm all { get { return redm.all; } }
        protected static redm any { get { return redm.any; } }
        protected static redm uni { get { return redm.uni; } }

        protected static scale shr7 { get { return scale.shr7; } }
        protected static scale shr15 { get { return scale.shr15; } }

        protected static space local { get { return space.local; } }
        protected static space shared { get { return space.shared; } }
        protected static space global { get { return space.global; } }
        protected static space param { get { return space.param; } }
        // note. use methods of spaceExtensions class to check whether some space refers to constant memory
        // it's all that complicated since there are 11 independent constant banks, so we cannot simply compare space to @const

        protected static squery surf_width { get { return squery.width; } }
        protected static squery surf_height { get { return squery.height; } }
        protected static squery surf_depth { get { return squery.depth; } }
        protected static squery surf_channel_datatype { get { return squery.channel_datatype; } }
        protected static squery surf_channel_order { get { return squery.channel_order; } }

        protected static test finite { get { return test.finite; } }
        protected static test infinite { get { return test.infinite; } }
        protected static test number { get { return test.number; } }
        protected static test notanumber { get { return test.notanumber; } }
        protected static test normal { get { return test.normal; } }
        protected static test subnormal { get { return test.subnormal; } }

        protected static tquery tex_width { get { return tquery.width; } }
        protected static tquery tex_height { get { return tquery.height; } }
        protected static tquery tex_depth { get { return tquery.depth; } }
        protected static tquery tex_channel_datatype { get { return tquery.channel_datatype; } }
        protected static tquery tex_channel_order { get { return tquery.channel_order; } }
        protected static tquery tex_normalized_coords { get { return tquery.normalized_coords; } }
        protected static tquery tex_filter_mode { get { return tquery.filter_mode; } }
        protected static tquery tex_addr_mode_0 { get { return tquery.addr_mode_0; } }
        protected static tquery tex_addr_mode_1 { get { return tquery.addr_mode_1; } }
        protected static tquery tex_addr_mode_2 { get { return tquery.addr_mode_2; } }

        protected static vshm vshm_clamp { get { return vshm.clamp; } }
        protected static vshm vshm_wrap { get { return vshm.wrap; } }

        #endregion

        #region Type checking rountines

        protected bool agree(Type wannabe, Type t)
        {
            return wannabe.agree(t);
        }

        protected bool agree(Expression expr, Type t)
        {
            return expr.agree(t);
        }

        protected bool agree_or_null(Type wannabe, Type t)
        {
            return wannabe.agree_or_null(t);
        }

        protected bool agree_or_null(Expression expr, Type t)
        {
            return expr.agree_or_null(t);
        }

        protected bool relaxed_agree(Type wannabe, Type t)
        {
            return wannabe.relaxed_agree(t);
        }

        protected bool relaxed_agree(Expression expr, Type t)
        {
            return expr.relaxed_agree(t);
        }

        protected bool relaxed_agree_or_null(Type wannabe, Type t)
        {
            return wannabe.relaxed_agree_or_null(t);
        }

        protected bool relaxed_agree_or_null(Expression expr, Type t)
        {
            return expr.relaxed_agree_or_null(t);
        }

        #endregion
    }
}