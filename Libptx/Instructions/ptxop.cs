using System;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Expressions;
using Libptx.Expressions.Immediate;
using Libptx.Expressions.Slots;
using Libptx.Expressions.Sregs;
using Libptx.Reflection;
using Libptx.Statements;
using XenoGears.Strings;
using Type = Libptx.Common.Types.Type;
using XenoGears.Functional;
using XenoGears.Assertions;

namespace Libptx.Instructions
{
    [DebuggerNonUserCode]
    public abstract partial class ptxop : Instruction
    {
        protected sealed override SoftwareIsa CustomVersion { get { return (SoftwareIsa)Math.Max((int)core_swisa, (int)custom_swisa); } }
        protected virtual SoftwareIsa custom_swisa { get { return SoftwareIsa.PTX_10; } }
        private SoftwareIsa core_swisa { get { return this.PtxopState().Affixes.OfType<Type>().MaxOrDefault(t => t.Version); } }

        protected sealed override HardwareIsa CustomTarget { get { return (HardwareIsa)Math.Max((int)core_hwisa, (int)custom_hwisa); } }
        protected virtual HardwareIsa custom_hwisa { get { return HardwareIsa.SM_10; } }
        private HardwareIsa core_hwisa { get { return this.PtxopState().Affixes.OfType<Type>().MaxOrDefault(t => t.Target); } }

        protected sealed override void CustomValidate()
        {
            base.CustomValidate();
            validate_opcode();
            custom_validate_opcode();
            validate_operands();
            custom_validate_operands();
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
        protected virtual void custom_validate_opcode() { }
        private void validate_opcode()
        {
            var state = this.PtxopState();
            state.Affixes.OfType<Type>().ForEach(t =>
            {
                t.AssertNotNull();
                t.Validate();

                (t.is_int() && t.bits() == 8).AssertImplies(allow_int8);
                (t.is_float() && t.bits() == 16).AssertImplies(allow_float16);
                (t.is_bit() && t.bits() == 8).AssertImplies(allow_bit8);
                (t.is_bit() && t.bits() == 16).AssertImplies(allow_bit16);
                (t.is_bit() && t.bits() == 32).AssertImplies(allow_bit32);
                (t.is_bit() && t.bits() == 64).AssertImplies(allow_bit64);
                t.is_pred().AssertImplies(allow_pred);
                t.is_vec().AssertImplies(allow_vec);
                t.is_arr().AssertImplies(allow_array);
            });
        }

        protected virtual void custom_validate_operands() { }
        private void validate_operands()
        {
            var arg_cnts = this.PtxopSigs().Select(sig => sig.Operands.Count());
            arg_cnts.Contains(Operands.Count()).AssertTrue();

            Operands.ForEach(arg =>
            {
                // this is commented out because operands may be optional
                // arg.AssertNotNull();

                if (arg != null) arg.Validate();
            });
        }

        protected override void RenderPtx() { Pragmas.ForEach(p => p.RenderPtx()); var ptx = core_render_ptx(); ptx = ptx ?? custom_render_ptx(ptx); writer.Write(ptx); }
        protected virtual String custom_render_ptx(String core) { return null; }
        private String core_render_ptx()
        {
            var buf = new StringBuilder();
            var meta = this.PtxopState();

            buf.Append(meta.Opcode);
            buf.Append(meta.Mods.Where(o => o != null).Select(o => o.Signature() ?? o.ToInvariantString()).StringJoin(""));
            buf.Append(meta.Affixes.Where(o => o != null).Select(o => o.Signature() ?? o.ToInvariantString()).StringJoin(", "));

            if (meta.Operands.IsNotEmpty()) buf.Append(" ");
            buf.Append(meta.Operands.Where(o => o != null).Select(o => o.RunRenderPtx()).StringJoin(", "));

            return buf.ToString();
        }

        protected override void RenderCubin()
        {
            throw new NotImplementedException();
        }

        #region Location checking utilities

        protected bool is_reg(Expression expr)
        {
            var ok = expr is Reg;
            ok |= (expr is Vector && ((Vector)expr).Flatten().All(is_reg));
            ok |= (expr is Modded && ((Modded)expr).Flatten().All(is_reg));
            return ok;
        }

        protected bool is_alu(Expression expr)
        {
            var ok = is_reg(expr) || expr is Const || expr is WarpSz;
            ok |= (expr is Vector && ((Vector)expr).Flatten().All(e => is_reg(e) || is_alu(e)));
            ok |= (expr is Modded && ((Modded)expr).Flatten().All(e => is_reg(e) || is_alu(e)));
            return ok;
        }

        protected bool is_alu_or_sreg(Expression expr)
        {
            var ok = is_alu(expr) || expr is Sreg;
            ok |= (expr is Vector && ((Vector)expr).Flatten().All(e => is_reg(e) || is_alu(e) || is_alu_or_sreg(e)));
            ok |= (expr is Modded && ((Modded)expr).Flatten().All(e => is_reg(e) || is_alu(e) || is_alu_or_sreg(e)));
            return ok;
        }

        #endregion

        #region Type checking utilities

        protected bool is_texref(Expression expr)
        {
            return expr.is_texref();
        }

        protected bool is_texref_or_null(Expression expr)
        {
            return expr == null || expr.is_texref();
        }

        protected bool is_samplerref(Expression expr)
        {
            return expr.is_samplerref();
        }

        protected bool is_samplerref_or_null(Expression expr)
        {
            return expr == null || expr.is_samplerref();
        }

        protected bool is_surfref(Expression expr)
        {
            return expr.is_surfref();
        }

        protected bool is_surfref_or_null(Expression expr)
        {
            return expr == null || expr.is_surfref();
        }

        protected bool is_ptr(Expression expr)
        {
            return expr.is_ptr();
        }

        protected bool is_ptr_or_null(Expression expr)
        {
            return expr == null || expr.is_ptr();
        }

        protected bool is_ptr(Expression expr, space space)
        {
            return expr.is_ptr(space);
        }

        protected bool is_ptr_or_null(Expression expr, space space)
        {
            return expr == null || expr.is_ptr(space);
        }

        protected bool is_bmk(Expression expr)
        {
            return expr.is_bmk();
        }

        protected bool is_bmk_or_null(Expression expr)
        {
            return expr == null || expr.is_bmk();
        }

        #endregion

        #region Operand checking utilities

        protected bool is_reg(Expression expr, Type t)
        {
            return is_reg(expr, t, 0);
        }

        protected bool is_reg(Expression expr, Type t, Mod mod)
        {
            if (expr == null) return false;
            if (!is_reg(expr)) return false;
            if (!expr.agree(t)) return false;
            if (!expr.has_mod(mod)) return false;
            return true;
        }

        protected bool is_reg_or_null(Expression expr, Type t)
        {
            return is_reg_or_null(expr, t, 0);
        }

        protected bool is_reg_or_null(Expression expr, Type t, Mod mod)
        {
            return expr == null || is_reg(expr, t, mod);
        }

        protected bool is_alu(Expression expr, Type t)
        {
            return is_alu(expr, t, 0);
        }

        protected bool is_alu(Expression expr, Type t, Mod mod)
        {
            if (expr == null) return false;
            if (!is_alu(expr)) return false;
            if (!expr.agree(t)) return false;
            if (!expr.has_mod(mod)) return false;
            return true;
        }

        protected bool is_alu_or_null(Expression expr, Type t)
        {
            return is_alu_or_null(expr, t, 0);
        }

        protected bool is_alu_or_null(Expression expr, Type t, Mod mod)
        {
            return expr == null || is_alu(expr, t, mod);
        }

        protected bool is_alu_or_sreg(Expression expr, Type t)
        {
            return is_alu_or_sreg(expr, t, 0);
        }

        protected bool is_alu_or_sreg(Expression expr, Type t, Mod mod)
        {
            if (expr == null) return false;
            if (!is_alu_or_sreg(expr)) return false;
            if (!expr.agree(t)) return false;
            if (!expr.has_mod(mod)) return false;
            return true;
        }

        protected bool is_alu_or_sreg_or_null(Expression expr, Type t)
        {
            return is_alu_or_sreg_or_null(expr, t, 0);
        }

        protected bool is_alu_or_sreg_or_null(Expression expr, Type t, Mod mod)
        {
            return expr == null || is_alu_or_sreg(expr, t, mod);
        }

        protected bool is_relaxed_reg(Expression expr, Type t)
        {
            return is_relaxed_reg(expr, t, 0);
        }

        protected bool is_relaxed_reg(Expression expr, Type t, Mod mod)
        {
            if (expr == null) return false;
            if (!is_reg(expr)) return false;
            if (!expr.relaxed_agree(t)) return false;
            if (!expr.has_mod(mod)) return false;
            return true;
        }

        protected bool is_relaxed_reg_or_null(Expression expr, Type t)
        {
            return is_relaxed_reg_or_null(expr, t, 0);
        }

        protected bool is_relaxed_reg_or_null(Expression expr, Type t, Mod mod)
        {
            return expr == null || is_relaxed_reg(expr, t, mod);
        }

        protected bool is_relaxed_alu(Expression expr, Type t)
        {
            return is_relaxed_alu(expr, t, 0);
        }

        protected bool is_relaxed_alu(Expression expr, Type t, Mod mod)
        {
            if (expr == null) return false;
            if (!is_alu(expr)) return false;
            if (!expr.relaxed_agree(t)) return false;
            if (!expr.has_mod(mod)) return false;
            return true;
        }

        protected bool is_relaxed_alu_or_null(Expression expr, Type t)
        {
            return is_relaxed_alu_or_null(expr, t, 0);
        }

        protected bool is_relaxed_alu_or_null(Expression expr, Type t, Mod mod)
        {
            return expr == null || is_relaxed_alu(expr, t, mod);
        }

        protected bool is_relaxed_alu_or_sreg(Expression expr, Type t)
        {
            return is_relaxed_alu_or_sreg(expr, t, 0);
        }

        protected bool is_relaxed_alu_or_sreg(Expression expr, Type t, Mod mod)
        {
            if (expr == null) return false;
            if (!is_alu_or_sreg(expr)) return false;
            if (!expr.relaxed_agree(t)) return false;
            if (!expr.has_mod(mod)) return false;
            return true;
        }

        protected bool is_relaxed_alu_or_sreg_or_null(Expression expr, Type t)
        {
            return is_relaxed_alu_or_sreg_or_null(expr, t, 0);
        }

        protected bool is_relaxed_alu_or_sreg_or_null(Expression expr, Type t, Mod mod)
        {
            return expr == null || is_relaxed_alu_or_sreg(expr, t, mod);
        }

        #endregion
    }
}