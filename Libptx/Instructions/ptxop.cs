using System;
using System.Diagnostics;
using System.IO;
using Libcuda.Versions;
using Libptx.Common.Spaces;
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

        protected virtual void custom_validate_operands(Module ctx) { }
        private void validate_operands(Module ctx)
        {
            // todo. don't verify nullity - custom validation will uncover those if they are present
            // todo. also verify count of Operands collection!
            // todo. also run validate for all operands
            // todo. but keep in mind that operands might be null so check for nullity!
            throw new NotImplementedException();
        }

        #region Operand checking utilities

        protected static Mod not { get { return Mod.Not; } }
        protected static Mod couple { get { return Mod.Couple; } }
        protected static Mod neg { get { return Mod.Neg; } }
        protected static Mod sel { get { return Mod.B0 | Mod.B1 | Mod.B2 | Mod.B3 | Mod.H0 | Mod.H1; } }
        protected static Mod member { get { return Mod.X | Mod.R | Mod.Y | Mod.G | Mod.Z | Mod.B | Mod.W | Mod.A; } }
        protected static Mod exact(Mod mod) { return mod | (Mod)65536; }

        protected bool is_alu(Expression expr, Type t)
        {
            return is_alu(expr, t, 0);
        }

        protected bool is_alu(Expression expr, Type t, Mod mod)
        {
            // todo. verify types of expressions using agree(expr.Type, t)
            // todo. only allow immediates (Const, Vector, WarpSz) and register vars (Var, Modded with Base == register Var)
            // todo. disallow specials here!
            // todo. if mod == 0, then disallow mods
            // todo. if mod != 0, then var might lack it and it'll be ok
            // todo. check mod using Flags!
            // todo. correctly process exact(mod)!
            throw new NotImplementedException();
        }

        protected bool is_alu_or_null(Expression expr, Type t)
        {
            return is_alu_or_null(expr, t, 0);
        }

        protected bool is_alu_or_null(Expression expr, Type t, Mod mod)
        {
            return expr == null || is_alu(expr, t, mod);
        }

        protected bool is_reg(Expression expr, Type t)
        {
            return is_reg(expr, t, 0);
        }

        protected bool is_reg(Expression expr, Type t, Mod mod)
        {
            // todo. same as for is_alu, but disallow immediates
            throw new NotImplementedException();
        }

        protected bool is_reg_or_null(Expression expr, Type t)
        {
            return is_reg_or_null(expr, t, 0);
        }

        protected bool is_reg_or_null(Expression expr, Type t, Mod mod)
        {
            return expr == null || is_reg(expr, t, mod);
        }

        protected bool is_special(Expression expr, Type t)
        {
            return is_special(expr, t, 0);
        }

        protected bool is_special(Expression expr, Type t, Mod mod)
        {
            // todo. same as for is_alu, but allow specials
            // todo. exception for types of special registers that specify grid characteristics
            throw new NotImplementedException();
        }

        protected bool is_special_or_null(Expression expr, Type t)
        {
            return is_special_or_null(expr, t, 0);
        }

        protected bool is_special_or_null(Expression expr, Type t, Mod mod)
        {
            return expr == null || is_special(expr, t, mod);
        }

        protected bool is_relaxed_alu(Expression expr, Type t)
        {
            return is_relaxed_alu(expr, t, 0);
        }

        protected bool is_relaxed_alu(Expression expr, Type t, Mod mod)
        {
            // todo. same as for is_alu, but with the use of relaxed_agree
            throw new NotImplementedException();
        }

        protected bool is_relaxed_alu_or_null(Expression expr, Type t)
        {
            return is_relaxed_alu_or_null(expr, t, 0);
        }

        protected bool is_relaxed_alu_or_null(Expression expr, Type t, Mod mod)
        {
            return expr == null || is_relaxed_alu(expr, t, mod);
        }

        protected bool is_relaxed_reg(Expression expr, Type t)
        {
            return is_relaxed_reg(expr, t, 0);
        }

        protected bool is_relaxed_reg(Expression expr, Type t, Mod mod)
        {
            // todo. same as for is_reg, but with the use of relaxed_agree
            throw new NotImplementedException();
        }

        protected bool is_relaxed_reg_or_null(Expression expr, Type t)
        {
            return is_relaxed_reg_or_null(expr, t, 0);
        }

        protected bool is_relaxed_reg_or_null(Expression expr, Type t, Mod mod)
        {
            return expr == null || is_relaxed_reg(expr, t, mod);
        }

        protected bool is_relaxed_special(Expression expr, Type t)
        {
            return is_relaxed_special(expr, t, 0);
        }

        protected bool is_relaxed_special(Expression expr, Type t, Mod mod)
        {
            // todo. same as for is_special, but with the use of relaxed_agree
            throw new NotImplementedException();
        }

        protected bool is_relaxed_special_or_null(Expression expr, Type t)
        {
            return is_relaxed_special_or_null(expr, t, 0);
        }

        protected bool is_relaxed_special_or_null(Expression expr, Type t, Mod mod)
        {
            return expr == null || is_relaxed_special(expr, t, mod);
        }

        protected bool is_ptr(Expression expr, space space)
        {
            // todo. any pointer agrees with space == 0
            // todo. check space using Flags (see atom.cs for more info)
            throw new NotImplementedException();
        }

        protected bool is_ptr_or_null(Expression expr, space space)
        {
            return expr == null || is_ptr(expr, space);
        }

        #endregion
    }
}