using System;
using System.Diagnostics;
using System.IO;
using Libcuda.Versions;
using Libptx.Common.Enumerations;
using Libptx.Expressions;
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

        protected static Mod not { get { return Mod.Not; } }
        protected static Mod couple { get { return Mod.Couple; } }
        protected static Mod neg { get { return Mod.Neg; } }
        protected static Mod sel { get { return Mod.B0 | Mod.B1 | Mod.B2 | Mod.B3 | Mod.H0 | Mod.H1; } }
        protected static Mod member { get { return Mod.X | Mod.R | Mod.Y | Mod.G | Mod.Z | Mod.B | Mod.W | Mod.A; } }
        protected static Mod exact(Mod mod) { return mod | (Mod)65536; }

        protected bool agree(Expression expr, Type t)
        {
            return agree(expr, t, 0);
        }

        protected bool agree(Expression expr, Type t, Mod mod)
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

        protected bool agree_or_null(Expression expr, Type t, Mod mod)
        {
            return expr == null || agree(expr, t, mod);
        }

        protected bool relax(Expression expr, Type t)
        {
            return relax(expr, t, 0);
        }

        protected bool relax(Expression expr, Type t, Mod mod)
        {
            // todo. same as for agree
            throw new NotImplementedException();
        }

        protected bool is_reg(Expression expr)
        {
            // todo. inline vectors are always reg!
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