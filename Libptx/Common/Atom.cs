using System;
using System.Collections.Generic;
using System.IO;
using Libcuda.Versions;
using Libptx.Common.Enumerations;
using Libptx.Common.Annotations.Atoms;
using Libptx.Common.Types;
using Type=Libptx.Common.Types.Type;
using XenoGears.Assertions;

namespace Libptx.Common
{
    public abstract class Atom : Validatable, Renderable
    {
        private IList<Location> _locations = new List<Location>();
        public IList<Location> Locations
        {
            get { return _locations; }
            set { _locations = value ?? new List<Location>(); }
        }

        private IList<String> _pragmas = new List<String>();
        public IList<String> Pragmas
        {
            get { return _pragmas; }
            set { _pragmas = value ?? new List<String>(); }
        }

        public SoftwareIsa Version { get { return (SoftwareIsa)Math.Max((int)CoreVersion, (int)CustomVersion); } }
        protected SoftwareIsa CoreVersion { get { throw new NotImplementedException(); } }
        protected virtual SoftwareIsa CustomVersion { get { return SoftwareIsa.PTX_10; } }

        public HardwareIsa Target { get { return (HardwareIsa)Math.Max((int)CoreTarget, (int)CustomTarget); } }
        protected HardwareIsa CoreTarget { get { throw new NotImplementedException(); } }
        protected virtual HardwareIsa CustomTarget { get { return HardwareIsa.SM_10; } }

        public void Validate(Module ctx) { (ctx.Version >= Version).AssertTrue(); (ctx.Target >= Target).AssertTrue(); CustomValidate(ctx); }
        protected virtual void CustomValidate(Module ctx) { /* do nothing */ }

        void Renderable.RenderAsPtx(TextWriter writer) { RenderAsPtx(writer); }
        protected abstract void RenderAsPtx(TextWriter writer);
        public sealed override String ToString() { return this.RenderAsPtx(); }

        #region Enumeration values => Static properties

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
        protected static Type texref { get { return new Type { Name = TypeName.Texref }; } }
        protected static Type samplerref { get { return new Type { Name = TypeName.Samplerref }; } }
        protected static Type surfref { get { return new Type { Name = TypeName.Surfref }; } }
        protected static Type ptr { get { return new Type { Name = TypeName.Ptr }; } }

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

        protected static size sz32 { get { return size.sz32; } }
        protected static size sz64 { get { return size.sz64; } }

        protected static space reg { get { return space.reg; } }
        protected static space sreg { get { return space.sreg; } }
        protected static space local { get { return space.local; } }
        protected static space shared { get { return space.shared; } }
        protected static space global { get { return space.global; } }
        protected static space param { get { return space.param; } }
        protected static space @const { get { return space.@const; } }
        protected static space const0 { get { return space.const0; } }
        protected static space const1 { get { return space.const1; } }
        protected static space const2 { get { return space.const2; } }
        protected static space const3 { get { return space.const3; } }
        protected static space const4 { get { return space.const4; } }
        protected static space const5 { get { return space.const5; } }
        protected static space const6 { get { return space.const6; } }
        protected static space const7 { get { return space.const7; } }
        protected static space const8 { get { return space.const8; } }
        protected static space const9 { get { return space.const9; } }
        protected static space const10 { get { return space.const10; } }

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
    }
}