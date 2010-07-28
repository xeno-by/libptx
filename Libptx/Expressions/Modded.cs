using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Libptx.Common;
using Libptx.Expressions.Slots;
using Type=Libptx.Common.Types.Type;
using XenoGears.Assertions;
using XenoGears.Functional;

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

        protected override void CustomValidate(Module ctx)
        {
            // todo. verify the following hypotheses:
            // 1) only Reg expressions can be modded
            // 2) Const and Vector immediates cannot be modded
            // 3) members accesses (xyzw, rgba) cannot be modded

            (Type != null).AssertTrue();
            Type.Validate(ctx);

            (Expr == null).AssertImplies((Mod & Mod.Couple) == Mod.Couple);
            if (Expr != null) Expr.Validate(ctx);
            if (Expr != null) (Expr is Reg).AssertTrue();

            (Embedded.IsNotEmpty()).AssertImplies((Mod & Mod.Couple) == Mod.Couple);
            Embedded.ForEach(e => { e.AssertNotNull(); e.Validate(ctx); });

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

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }

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