using System;
using System.Diagnostics;
using System.IO;
using Libptx.Instructions.Enumerations;
using Libcuda.Versions;
using Libptx.Statements;

namespace Libptx.Instructions
{
    [DebuggerNonUserCode]
    public abstract class ptxop : Instruction
    {
        #region Enumeration values => Static properties

        protected static type u8 { get { return type.u8; } }
        protected static type s8 { get { return type.s8; } }
        protected static type u16 { get { return type.u16; } }
        protected static type s16 { get { return type.s16; } }
        protected static type u32 { get { return type.u32; } }
        protected static type s32 { get { return type.s32; } }
        protected static type u64 { get { return type.u64; } }
        protected static type s64 { get { return type.s64; } }
        protected static type f16 { get { return type.f32; } }
        protected static type f32 { get { return type.f32; } }
        protected static type f64 { get { return type.f64; } }
        protected static type b8 { get { return type.b8; } }
        protected static type b16 { get { return type.b16; } }
        protected static type b32 { get { return type.b32; } }
        protected static type b64 { get { return type.b64; } }

        protected static frnd rn { get { return frnd.rn; } }
        protected static frnd rz { get { return frnd.rz; } }
        protected static frnd rm { get { return frnd.rm; } }
        protected static frnd rp { get { return frnd.rp; } }

        protected static irnd rni { get { return irnd.rni; } }
        protected static irnd rzi { get { return irnd.rzi; } }
        protected static irnd rmi { get { return irnd.rmi; } }
        protected static irnd rpi { get { return irnd.rpi; } }

        protected static mulm mulm_hi { get { return mulm.hi; } }
        protected static mulm mulm_lo { get { return mulm.lo; } }
        protected static mulm wide { get { return mulm.wide; } }

        protected static prmtm f4e { get { return prmtm.f4e; } }
        protected static prmtm b4e { get { return prmtm.b4e; } }
        protected static prmtm rc8 { get { return prmtm.rc8; } }
        protected static prmtm ec1 { get { return prmtm.ec1; } }
        protected static prmtm ecr { get { return prmtm.ecr; } }
        protected static prmtm rc16 { get { return prmtm.rc16; } }

        protected static testpop finite { get { return testpop.finite; } }
        protected static testpop infinite { get { return testpop.infinite; } }
        protected static testpop number { get { return testpop.number; } }
        protected static testpop notanumber { get { return testpop.notanumber; } }
        protected static testpop normal { get { return testpop.normal; } }
        protected static testpop subnormal { get { return testpop.subnormal; } }

        protected static cmpop eq { get { return cmpop.eq; } }
        protected static cmpop ne { get { return cmpop.ne; } }
        protected static cmpop lt { get { return cmpop.lt; } }
        protected static cmpop le { get { return cmpop.le; } }
        protected static cmpop gt { get { return cmpop.gt; } }
        protected static cmpop ge { get { return cmpop.ge; } }
        protected static cmpop cmpop_lo { get { return cmpop.lo; } }
        protected static cmpop ls { get { return cmpop.ls; } }
        protected static cmpop cmpop_hi { get { return cmpop.hi; } }
        protected static cmpop hs { get { return cmpop.hs; } }
        protected static cmpop equ { get { return cmpop.equ; } }
        protected static cmpop neu { get { return cmpop.neu; } }
        protected static cmpop ltu { get { return cmpop.ltu; } }
        protected static cmpop leu { get { return cmpop.leu; } }
        protected static cmpop gtu { get { return cmpop.gtu; } }
        protected static cmpop geu { get { return cmpop.geu; } }
        protected static cmpop num { get { return cmpop.num; } }
        protected static cmpop nan { get { return cmpop.nan; } }

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

        protected static cop ca { get { return cop.ca; } }
        protected static cop cg { get { return cop.cg; } }
        protected static cop cs { get { return cop.cs; } }
        protected static cop lu { get { return cop.lu; } }
        protected static cop cv { get { return cop.cv; } }
        protected static cop wb { get { return cop.wb; } }
        protected static cop wt { get { return cop.wt; } }

        protected static vec v2 { get { return vec.v2; } }
        protected static vec v4 { get { return vec.v4; } }

        protected static ss @const { get { return ss.@const; } }
        protected static ss global { get { return ss.global; } }
        protected static ss local { get { return ss.local; } }
        protected static ss param { get { return ss.param; } }
        protected static ss shared { get { return ss.shared; } }

        protected static cachelevel L1 { get { return cachelevel.L1; } }
        protected static cachelevel L2 { get { return cachelevel.L2; } }

        protected static size sz32 { get { return size.sz32; } }
        protected static size sz64 { get { return size.sz64; } }

        protected static geom d1 { get { return geom.d1; } }
        protected static geom d2 { get { return geom.d2; } }
        protected static geom d3 { get { return geom.d3; } }

        protected static tquery tex_width { get { return tquery.width; } }
        protected static tquery tex_height { get { return tquery.height; } }
        protected static tquery tex_depth { get { return tquery.depth; } }
        protected static tquery tex_channel_datatype { get { return tquery.channel_datatype; } }
        protected static tquery tex_channel_order { get { return tquery.channel_order; } }
        protected static tquery tex_normalized_coords { get { return tquery.normalized_coords; } }

        protected static tquerys tex_filter_mode { get { return tquerys.filter_mode; } }
        protected static tquerys tex_addr_mode_0 { get { return tquerys.addr_mode_0; } }
        protected static tquerys tex_addr_mode_1 { get { return tquerys.addr_mode_1; } }
        protected static tquerys tex_addr_mode_2 { get { return tquerys.addr_mode_2; } }

        protected static clampm clamp_trap { get { return clampm.trap; } }
        protected static clampm clamp_clamp { get { return clampm.clamp; } }
        protected static clampm clamp_zero { get { return clampm.zero; } }

        protected static squery surf_width { get { return squery.width; } }
        protected static squery surf_height { get { return squery.height; } }
        protected static squery surf_depth { get { return squery.depth; } }
        protected static squery surf_channel_datatype { get { return squery.channel_datatype; } }
        protected static squery surf_channel_order { get { return squery.channel_order; } }

        protected static barlevel cta { get { return barlevel.cta; } }
        protected static barlevel gl { get { return barlevel.gl; } }
        protected static barlevel sys { get { return barlevel.sys; } }

        protected static sel b0 { get { return sel.b0; } }
        protected static sel b1 { get { return sel.b1; } }
        protected static sel b2 { get { return sel.b2; } }
        protected static sel b3 { get { return sel.b3; } }
        protected static sel h0 { get { return sel.h0; } }
        protected static sel h1 { get { return sel.h1; } }

        protected static vshm vshm_clamp { get { return vshm.clamp; } }
        protected static vshm vshm_wrap { get { return vshm.wrap; } }

        protected static scale shr7 { get { return scale.shr7; } }
        protected static scale shr15 { get { return scale.shr15; } }

        #endregion

        protected override SoftwareIsa CustomVersion { get { return (SoftwareIsa)Math.Max((int)custom_swisa, (int)default_swisa); } }
        protected virtual SoftwareIsa custom_swisa { get { return SoftwareIsa.PTX_10; } }
        private SoftwareIsa default_swisa
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        protected override HardwareIsa CustomTarget { get { return (HardwareIsa)Math.Max((int)custom_hwisa, (int)default_hwisa); } }
        protected virtual HardwareIsa custom_hwisa { get { return HardwareIsa.SM_10; } }
        private HardwareIsa default_hwisa
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        protected virtual bool allow_int8 { get { return false; } }
        protected virtual bool allow_float16 { get { return false; } }
        protected virtual bool allow_bit8 { get { return false; } }
        protected virtual bool allow_bit16 { get { return false; } }
        protected virtual bool allow_bit32 { get { return false; } }
        protected virtual bool allow_bit64 { get { return false; } }
        protected virtual bool allow_pred { get { return false; } }
        protected override void CustomValidate(Module ctx) { custom_validate_opcode(ctx.Version, ctx.Target); custom_validate_op(ctx.Version, ctx.Target); }
        protected virtual void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa) {}
        protected virtual void custom_validate_op(SoftwareIsa target_swisa, HardwareIsa target_hwisa) {}

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}