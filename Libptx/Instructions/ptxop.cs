using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Libcuda.Versions;
using Libptx.Common.Annotations;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Common.Types.Opaques;
using Libptx.Common.Types.Pointers;
using Libptx.Expressions;
using Libptx.Expressions.Immediate;
using Libptx.Expressions.Slots;
using Libptx.Expressions.Sregs;
using Libptx.Statements;
using XenoGears.Reflection.Shortcuts;
using Type = Libptx.Common.Types.Type;
using XenoGears.Reflection.Attributes;
using XenoGears.Functional;
using XenoGears;
using XenoGears.Assertions;

namespace Libptx.Instructions
{
    [DebuggerNonUserCode]
    public abstract partial class ptxop : Instruction
    {
        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }

        protected sealed override SoftwareIsa CustomVersion { get { return (SoftwareIsa)Math.Max((int)custom_swisa, (int)default_swisa); } }
        protected virtual SoftwareIsa custom_swisa { get { return SoftwareIsa.PTX_10; } }
        private SoftwareIsa default_swisa
        {
            get
            {
                var t_swisa = this.Version();

                var props = this.GetType().GetProperties(BF.PublicInstance | BF.DeclOnly).Where(p => p.HasAttr<ParticleAttribute>()).ToReadOnly();
                var p_swisas = props.Select(p =>
                {
                    var v = p.GetValue(this, null);
                    var @default = p.PropertyType.Fluent(t => t.IsValueType ? Activator.CreateInstance(t) : null);
                    return Equals(v, @default) ? 0 : p.Version();
                }).ToReadOnly();

                return (SoftwareIsa)Math.Max((int)t_swisa, (int)p_swisas.MaxOrDefault());
            }
        }

        protected sealed override HardwareIsa CustomTarget { get { return (HardwareIsa)Math.Max((int)custom_hwisa, (int)default_hwisa); } }
        protected virtual HardwareIsa custom_hwisa { get { return HardwareIsa.SM_10; } }
        private HardwareIsa default_hwisa
        {
            get
            {
                var t_hwisa = this.Version();

                var props = this.GetType().GetProperties(BF.PublicInstance | BF.DeclOnly).Where(p => p.HasAttr<ParticleAttribute>()).ToReadOnly();
                var p_hwisas = props.Select(p =>
                {
                    var v = p.GetValue(this, null);
                    var @default = p.PropertyType.Fluent(t => t.IsValueType ? Activator.CreateInstance(t) : null);
                    return Equals(v, @default) ? 0 : p.Target();
                }).ToReadOnly();

                return (HardwareIsa)Math.Max((int)t_hwisa, (int)p_hwisas.MaxOrDefault());
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
            var p_types = this.GetType().GetProperties(BF.PublicInstance | BF.DeclOnly).Where(p => p.PropertyType == typeof(Type)).ToReadOnly();
            p_types.ForEach(p =>
            {
                var t = (Type)p.GetValue(this, null);

                t.AssertNotNull();
                t.Validate(ctx);

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

        protected virtual void custom_validate_operands(Module ctx) { }
        private void validate_operands(Module ctx)
        {
            var arg_cnts = this.Signatures().Select(sig =>
            {
                var bal = 0;
                var commas = 0;
                sig.ForEach(c =>
                {
                    if (c == '{') bal++;
                    if (c == '}') bal--;
                    if (bal == 0 && c == ',') commas++;
                });

                return commas == 0 ? 0 : (commas + 1);
            }).ToReadOnly();
            arg_cnts.Contains(Operands.Count()).AssertTrue();

            Operands.ForEach(arg =>
            {
                // don't verify nullity - custom validation will uncover it if it's an error
                if (arg != null)
                {
                    arg.Validate(ctx);
                }
            });
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