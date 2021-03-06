using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using Libcuda.DataTypes;
using Libcuda.Versions;
using Libptx.Common.Types.Bits;
using Libptx.Expressions.Slots;
using Libptx.Reflection;
using Libptx.Statements;
using XenoGears.Assertions;
using XenoGears.Functional;
using XenoGears.Strings;
using ClrType = System.Type;
using XenoGears;

namespace Libptx.Common.Types
{
    [DebuggerNonUserCode]
    public partial class Type : Atom, IEquatable<Type>
    {
        public TypeName Name { get; set; }
        public TypeMod Mod { get; set; }
        public int[] Dims { get; set; }

        protected override SoftwareIsa CustomVersion { get { return Name.Version(); } }
        protected override HardwareIsa CustomTarget
        {
            get
            {
                if (Name == f64)
                {
                    var incapable_target = ctx.Target < HardwareIsa.SM_13;
                    var downgrade_doubles = ctx.DowngradeDoubles;
                    return incapable_target && !downgrade_doubles ? HardwareIsa.SM_13 : HardwareIsa.SM_10;
                }
                else
                {
                    return Name.Target();
                }
            }
        }

        protected override void CustomValidate()
        {
            (SizeOfElement <= 128 / 8).AssertTrue();

            this.is_vec().AssertImplies(this.el().is_scalar());
            this.vec_rank().AssertThat(rank => rank == 0 || rank == 2 || rank == 4);

            this.is_arr().AssertImplies(this.el().is_scalar());
            (Dims ?? new int[0]).AssertEach(dim => dim > 0);
        }

        protected override void RenderPtx()
        {
            var el = this.arr_el() ?? this;
            if (el.is_vec())
            {
                writer.Write(".v{0}", el.vec_rank());

                var aint_need_whitespace = ctx.Parent is Instruction;
                if (!aint_need_whitespace) writer.Write(" ");
            }

            writer.Write("." + Name.Signature().AssertNotNull());
            if (this.is_arr()) writer.Write(" " + (Dims ?? Seq.Empty<int>()).Select(dim => 
                dim == 0 ? "[]" : String.Format("[{0}]", dim)).StringJoin(String.Empty));
        }

        protected override void RenderCubin()
        {
            throw new NotImplementedException();
        }

        public int SizeInMemory { get { return SizeOfElement * (Dims ?? new int[0]).Product(); } }
        public int SizeOfElement
        {
            get
            {
                if (this.is_opaque()) return 0;
                if (this.is_pred()) return 0;
                if (this.is_ptr()) return 0;
                if (this.is_bmk()) return 0;

                var atom = this.el();
                var cnt = Math.Max((this.arr_el() ?? this).vec_rank(), 1);
                return cnt * Marshal.SizeOf((ClrType)atom);
            }
        }

        // public static implicit operator TypeName(Type t) { return t == null ? 0 : t.is_scalar() ? t.Name : 0; }
        public static implicit operator Type(TypeName t) { return t == 0 ? null : new Type { Name = t, Mod = TypeMod.Scalar }; }

        public ClrType ClrType { get { return (ClrType)this; } }
        public static implicit operator ClrType(Type t)
        {
            if (t == null) return null;
            var el = t.arr_el() ?? t;

            var clr = t.Name.ClrType();
            if (el.is_vec())
            {
                var rank = el.vec_rank();
                var tv1_name = typeof(int3).Namespace + "." + clr.GetCSharpRef(ToCSharpOptions.Terse) + rank;
                var tv1 = typeof(int3).Assembly.GetType(tv1_name);
                var tv2_name = typeof(Bit8).Namespace + "." + clr.GetCSharpRef(ToCSharpOptions.Terse) + rank;
                var tv2 = typeof(Bit8).Assembly.GetType(tv2_name);
                var tv3_name = typeof(Bit8).Namespace + "." + clr.GetCSharpRef(ToCSharpOptions.Terse) + "_V" + rank;
                var tv3 = typeof(Bit8).Assembly.GetType(tv3_name);
                clr = tv1 ?? tv2 ?? tv3;
            }

            if (t.is_arr())
            {
                var rank = t.arr_rank();
                clr = 1.UpTo(rank).Fold(clr, (aux, _) => aux == null ? null : aux.MakeArrayType());
            }

            return clr;
        }

        private static readonly Dictionary<ClrType, Type> pool = new Dictionary<ClrType, Type>();
        static Type()
        {
            var scalars = Enum.GetValues(typeof(TypeName)).Cast<TypeName>();
            var combos = Combinatorics.CartesianProduct(scalars, new []{ TypeMod.Scalar, TypeMod.V1, TypeMod.V2, TypeMod.V4 }, 0.UpTo(1));
            var types = combos.Zip((t, mod, dims) => new Type{Name = t, Mod = mod | (dims == 1 ? TypeMod.Array : TypeMod.Scalar), Dims = dims.Times(0).ToArray()});
            types.ForEach(t =>
            {
                var clr = t.ClrType;
                if (clr != null) pool.Add(t, t);
            });
        }

        public static implicit operator Type(ClrType t)
        {
            return pool.GetOrDefault(t);
        }

        public bool Equals(Type other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            var dim_equal = Seq.Equal(other.Dims ?? Seq.Empty<int>(), Dims ?? Seq.Empty<int>());
            return Equals(other.Name, Name) && Equals(other.Mod, Mod) && dim_equal;
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != typeof(Type)) return false;
            return Equals((Type)obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int result = Name.GetHashCode();
                result = (result * 397) ^ Mod.GetHashCode();
                result = (result * 397) ^ (Dims ?? Seq.Empty<int>()).Fold(0, (acc, curr) => acc ^ (curr * 397));
                return result;
            }
        }

        public static bool operator ==(Type left, Type right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(Type left, Type right)
        {
            return !Equals(left, right);
        }
    }
}