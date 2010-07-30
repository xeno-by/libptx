using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Expressions.Slots;
using Libptx.Expressions.Sregs;
using Type=Libptx.Common.Types.Type;
using XenoGears.Assertions;
using XenoGears.Functional;
using Libptx.Reflection;

namespace Libptx.Expressions
{
    [DebuggerNonUserCode]
    public class Modded : Atom, Expression
    {
        public Expression Expr { get; set; }
        public Mod Mod { get; set; }

        private IList<Expression> _embedded = new List<Expression>();
        public IList<Expression> Embedded
        {
            get { return _embedded; }
            set { _embedded = value ?? new List<Expression>(); }
        }

        public Type Type
        {
            get
            {
                if ((Mod & Mod.Not) == Mod.Not)
                {
                    return typeof(bool);
                }
                else if ((Mod & Mod.Couple) == Mod.Couple)
                {
                    return typeof(bool);
                }
                else if ((Mod & Mod.Neg) == Mod.Neg || (Mod & Mod.H0) == Mod.H0 || (Mod & Mod.H1) == Mod.H1 ||
                    (Mod & Mod.B0) == Mod.B0 || (Mod & Mod.B1) == Mod.B1 || (Mod & Mod.B2) == Mod.B2 || (Mod & Mod.B3) == Mod.B3)
                {
                    return Expr == null ? null : Expr.Type;
                }
                else if ((Mod & Mod.X) == Mod.X || (Mod & Mod.Y) == Mod.Y || (Mod & Mod.Z) == Mod.Z || (Mod & Mod.W) == Mod.W ||
                    (Mod & Mod.R) == Mod.R || (Mod & Mod.G) == Mod.G || (Mod & Mod.B) == Mod.B || (Mod & Mod.A) == Mod.A)
                {
                    return Expr == null ? null : Expr.vec_el();
                }
                else
                {
                    throw AssertionHelper.Fail();
                }
            }
        }

        protected override SoftwareIsa CustomVersion
        {
            get
            {
                var expr_version = Expr.Version();
                var embedded_version = Embedded.MaxOrDefault(el => el.Version());
                return (SoftwareIsa)Math.Max((int)expr_version, (int)embedded_version);
            }
        }

        protected override HardwareIsa CustomTarget
        {
            get
            {
                var expr_target = Expr.Version();
                var embedded_target = Embedded.MaxOrDefault(el => el.Target());
                return (HardwareIsa)Math.Max((int)expr_target, (int)embedded_target);
            }
        }

        protected override void CustomValidate()
        {
            // todo. verify the following hypotheses:
            // 1) only Reg expressions can be modded
            // 2) Const and Vector immediates cannot be modded
            // 3) members accesses (xyzw, rgba) cannot be modded

            (Type != null).AssertTrue();
            Type.Validate();

            (Expr == null).AssertImplies((Mod & Mod.Couple) == Mod.Couple);
            if (Expr != null) Expr.Validate();
            if (Expr != null) (Expr is Reg || Expr is Sreg).AssertTrue();

            (Embedded.IsNotEmpty()).AssertImplies((Mod & Mod.Couple) == Mod.Couple);
            Embedded.ForEach(e => { e.AssertNotNull(); e.Validate(); });

            if ((Mod & Mod.Not) == Mod.Not)
            {
                Expr.is_pred().AssertTrue();
                (Mod == Mod.Not).AssertTrue();
            }

            if ((Mod & Mod.Couple) == Mod.Couple)
            {
                (Expr == null).AssertTrue();
                (Mod == Mod.Couple).AssertTrue();
                (Embedded.Count() == 2).AssertTrue();
                Embedded.AssertEach(e => e.is_pred().AssertTrue());
            }

            if ((Mod & Mod.Neg) == Mod.Neg)
            {
                (agree(Expr, u32) || agree(Expr, s32)).AssertTrue();
                ((Mod & ~(Mod.Neg | Mod.B0 | Mod.B1 | Mod.B2 | Mod.B3 | Mod.H0 | Mod.H1)) == 0).AssertTrue();
            }

            if ((Mod & Mod.B0) == Mod.B0)
            {
                (agree(Expr, u32) || agree(Expr, s32)).AssertTrue();
                (Mod == Mod.B0 || Mod == (Mod.Neg | Mod.B0)).AssertTrue();
            }

            if ((Mod & Mod.B1) == Mod.B1)
            {
                (agree(Expr, u32) || agree(Expr, s32)).AssertTrue();
                (Mod == Mod.B1 || Mod == (Mod.Neg | Mod.B1)).AssertTrue();
            }

            if ((Mod & Mod.B2) == Mod.B2)
            {
                (agree(Expr, u32) || agree(Expr, s32)).AssertTrue();
                (Mod == Mod.B2 || Mod == (Mod.Neg | Mod.B2)).AssertTrue();
            }

            if ((Mod & Mod.B3) == Mod.B3)
            {
                (agree(Expr, u32) || agree(Expr, s32)).AssertTrue();
                (Mod == Mod.B3 || Mod == (Mod.Neg | Mod.B3)).AssertTrue();
            }

            if ((Mod & Mod.H0) == Mod.H0)
            {
                (agree(Expr, u32) || agree(Expr, s32)).AssertTrue();
                (Mod == Mod.H0 || Mod == (Mod.Neg | Mod.H0)).AssertTrue();
            }

            if ((Mod & Mod.H1) == Mod.H1)
            {
                (agree(Expr, u32) || agree(Expr, s32)).AssertTrue();
                (Mod == Mod.H1 || Mod == (Mod.Neg | Mod.H1)).AssertTrue();
            }

            if ((Mod & Mod.X) == Mod.X)
            {
                (Expr.vec_rank() >= 1).AssertTrue();
                (Mod == Mod.X).AssertTrue();
            }

            if ((Mod & Mod.R) == Mod.R)
            {
                (Expr.vec_rank() >= 1).AssertTrue();
                (Mod == Mod.R).AssertTrue();
            }

            if ((Mod & Mod.Y) == Mod.Y)
            {
                (Expr.vec_rank() >= 2).AssertTrue();
                (Mod == Mod.Y).AssertTrue();
            }

            if ((Mod & Mod.G) == Mod.G)
            {
                (Expr.vec_rank() >= 2).AssertTrue();
                (Mod == Mod.G).AssertTrue();
            }

            if ((Mod & Mod.Z) == Mod.Z)
            {
                (Expr.vec_rank() >= 3).AssertTrue();
                (Mod == Mod.Z).AssertTrue();
            }

            if ((Mod & Mod.B) == Mod.B)
            {
                (Expr.vec_rank() >= 3).AssertTrue();
                (Mod == Mod.B).AssertTrue();
            }

            if ((Mod & Mod.W) == Mod.W)
            {
                (Expr.vec_rank() >= 4).AssertTrue();
                (Mod == Mod.W).AssertTrue();
            }

            if ((Mod & Mod.A) == Mod.A)
            {
                (Expr.vec_rank() >= 4).AssertTrue();
                (Mod == Mod.A).AssertTrue();
            }
        }

        protected override void RenderPtx()
        {
            if (this.has_mod(not))
            {
                writer.Write(not.Signature());
                writer.Write(Expr);
            }
            else if (this.has_mod(couple))
            {
                writer.Write(Embedded.First());
                writer.Write(couple.Signature());
                writer.Write(Embedded.Second());
            }
            else if (this.has_mod(neg | sel))
            {
                if (this.has_mod(neg)) writer.Write("-");
                writer.Write(Expr);
                if (this.has_mod(sel))
                {
                    var postfix = (Mod & sel).Signature();
                    writer.Write("." + postfix);
                }
            }
            else if (this.has_mod(member))
            {
                writer.Write(Expr);

                var postfix = (Mod & member).Signature();
                writer.Write("." + postfix);
            }
            else
            {
                throw AssertionHelper.Fail();
            }
        }

        protected override void RenderCubin()
        {
            throw new NotImplementedException();
        }
    }

    [DebuggerNonUserCode]
    public static class ModdedExtensions
    {
        public static ReadOnlyCollection<Expression> Flatten(this Modded modded)
        {
            var lvl1 = modded.Expr.Concat(modded.Embedded).Where(e => e != null);
            var lvls_all = lvl1.SelectMany(e => e is Modded ? ((Modded)e).Flatten() : e.MkArray().ToReadOnly()).ToReadOnly();
            return lvls_all;
        }
    }
}