using System;
using System.Diagnostics;
using System.IO;
using Libcuda.Versions;
using Libptx.Common.Spaces;
using Libptx.Expressions;
using Libptx.Expressions.Slots;
using Libptx.Statements;
using Type = Libptx.Common.Types.Type;

namespace Libptx.Instructions
{
    [DebuggerNonUserCode]
    public abstract partial class ptxop : Instruction
    {
        public String Name { get { throw new NotImplementedException(); } }
        protected override void RenderAsPtx(TextWriter writer) { throw new NotImplementedException(); }

        protected sealed override SoftwareIsa CustomVersion { get { return (SoftwareIsa)Math.Max((int)custom_swisa, (int)default_swisa); } }
        protected virtual SoftwareIsa custom_swisa { get { return SoftwareIsa.PTX_10; } }
        private SoftwareIsa default_swisa
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        protected sealed override HardwareIsa CustomTarget { get { return (HardwareIsa)Math.Max((int)custom_hwisa, (int)default_hwisa); } }
        protected virtual HardwareIsa custom_hwisa { get { return HardwareIsa.SM_10; } }
        private HardwareIsa default_hwisa
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        protected sealed override void CustomValidate(Module ctx)
        {
            base.CustomValidate(ctx);
            validate_opcode(ctx);
            custom_validate_opcode(ctx);
            validate_operands(ctx);
            custom_validate_operands(ctx);
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
        protected virtual void custom_validate_opcode(Module ctx) { }
        private void validate_opcode(Module ctx)
        {
            // todo. verify nullity vs optional types?!
            // todo. verify all restrictions for types (i mean, allow_*)
            throw new NotImplementedException();
        }

        protected virtual bool allow_ptr { get { return false; } }
        protected virtual bool allow_special { get { return false; } }
        protected virtual void custom_validate_operands(Module ctx) { }
        private void validate_operands(Module ctx)
        {
            // todo. don't verify nullity - custom validation will uncover those if they are present
            // todo. verify all restrictions for operands (i mean, allow_*)
            // todo. constants are treated as regs
            // todo. also verify count of Operands collection!
            throw new NotImplementedException();
        }
    }
}