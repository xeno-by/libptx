using System;
using System.Diagnostics;
using System.IO;
using Libcuda.Versions;
using Libptx.Common.Enumerations;
using Libptx.Expressions;
using Libptx.Statements;

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
            // todo. all types must be not null
            // todo. now what about optional types?
            // todo. verify all restrictions for types (i mean, allow_*)
            throw new NotImplementedException();
        }

        protected virtual bool allow_ptr { get { return false; } }
        protected virtual bool allow_special { get { return false; } }
        protected virtual void custom_validate_operands(Module ctx) { }
        private void validate_operands(Module ctx)
        {
            // todo. all operands must be not null
            // todo. now what about optional operands?
            // todo. verify all restrictions for operands (i mean, allow_*)
            // todo. constants are treated as regs
            throw new NotImplementedException();
        }

        protected static VarMod not { get { return VarMod.Not; } }
        protected static VarMod couple { get { return VarMod.Couple; } }
        protected static VarMod neg { get { return VarMod.Neg; } }
        protected static VarMod sel { get { return VarMod.B0 | VarMod.B1 | VarMod.B2 | VarMod.B3 | VarMod.H0 | VarMod.H1; } }
        protected static VarMod member { get { return VarMod.X | VarMod.R | VarMod.Y | VarMod.G | VarMod.Z | VarMod.B | VarMod.W | VarMod.A; } }
        protected static VarMod exact(VarMod mod) { return mod | (VarMod)65536; }

        protected bool agree(Expression expr, Type t)
        {
            return agree(expr, t, 0);
        }

        protected bool agree(Expression expr, Type t, VarMod mod)
        {
            // todo. if mod == 0, then disallow mods
            // todo. allow instances of Vector!
            // todo. if mod != 0, then verify types of vars
            // todo. if mod != 0, then var might lack it and it'll be ok
            // todo. correctly process exact(mod)!
            // todo. also verify types of vars, e.g. couple can consist only of two preds
            // todo. exception for specials!
            throw new NotImplementedException();
        }

        protected bool agree_or_null(Expression expr, Type t)
        {
            return agree_or_null(expr, t, 0);
        }

        protected bool agree_or_null(Expression expr, Type t, VarMod mod)
        {
            return expr == null || agree(expr, t, mod);
        }

        protected bool relax(Expression expr, Type t)
        {
            return relax(expr, t, 0);
        }

        protected bool relax(Expression expr, Type t, VarMod mod)
        {
            // todo. same as for agree
            throw new NotImplementedException();
        }

        protected bool is_reg(Expression expr)
        {
            // todo. no mods are allowed here regardless of allow_XXX
            throw new NotImplementedException();
        }

        protected bool is_special(Expression expr)
        {
            throw new NotImplementedException();
        }

        protected static space code { get { return (space)65536; } }

        protected bool is_ptr(Expression expr, space space)
        {
            // todo. any pointer agrees with space == 0
            throw new NotImplementedException();
        }
    }
}