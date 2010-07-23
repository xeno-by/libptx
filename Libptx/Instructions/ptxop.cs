using System;
using System.Diagnostics;
using System.IO;
using Libcuda.Versions;
using Libptx.Statements;

namespace Libptx.Instructions
{
    [DebuggerNonUserCode]
    public abstract partial class ptxop : Instruction
    {
        public String Name { get { throw new NotImplementedException(); } }
        protected override void RenderAsPtx(TextWriter writer) { throw new NotImplementedException(); }

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
        protected virtual bool allow_array { get { return false; } }
        protected override void CustomValidate(Module ctx)
        {
            validate_opcode(ctx.Version, ctx.Target);
            custom_validate_opcode(ctx.Version, ctx.Target);
            validate_ops(ctx.Version, ctx.Target);
            custom_validate_ops(ctx.Version, ctx.Target);
        }

        protected virtual void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa) {}
        private void validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            // todo. all types must be not null
            throw new NotImplementedException();
        }

        protected virtual void custom_validate_ops(SoftwareIsa target_swisa, HardwareIsa target_hwisa) {}
        private void validate_ops(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            // todo. all operands must be not null
            throw new NotImplementedException();
        }
    }
}