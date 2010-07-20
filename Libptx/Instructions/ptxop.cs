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
        protected virtual bool allow_vec { get { return false; } }
        protected override void CustomValidate(Module ctx)
        {
            validate_opcode(ctx.Version, ctx.Target);
            custom_validate_opcode(ctx.Version, ctx.Target);
            validate_op(ctx.Version, ctx.Target);
            custom_validate_op(ctx.Version, ctx.Target);
        }

        protected virtual void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa) {}
        private void validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            throw new NotImplementedException();
        }

        protected virtual void custom_validate_op(SoftwareIsa target_swisa, HardwareIsa target_hwisa) {}
        private void validate_op(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            throw new NotImplementedException();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}