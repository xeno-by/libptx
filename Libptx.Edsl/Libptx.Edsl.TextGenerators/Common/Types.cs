using System;
using System.Collections.ObjectModel;
using System.Linq;
using Libptx.Common.Annotations;
using Libptx.Common.Types;
using XenoGears.Functional;
using Type = Libptx.Common.Types.Type;
using ClrType = System.Type;

namespace Libptx.Edsl.TextGenerators.Common
{
    internal static class Types
    {
        private static Context Context
        {
            get { return TextGenerators.Context.Current; }
        }

        public static ReadOnlyCollection<Type> Opaque
        {
            get
            {
                var all = Enum.GetValues(typeof(TypeName)).Cast<TypeName>().Select(t =>
                {
                    if (t.Version() > Context.Version) return null;
                    if (t.Target() > Context.Target) return null;
                    return (Type)t;
                }).Where(t => t != null).ToReadOnly();

                return all.Intersect(TypeName.Pred, TypeName.Texref, TypeName.Samplerref, TypeName.Surfref, TypeName.Ptr).ToReadOnly();
            }
        }

        public static ReadOnlyCollection<ClrType> OpaqueClr
        {
            get { return Opaque.Select(t => t.ClrType).ToReadOnly(); }
        }

        public static ReadOnlyCollection<Type> Scalar
        {
            get
            {
                var all = Enum.GetValues(typeof(TypeName)).Cast<TypeName>().Select(t =>
                {
                    if (t.Version() > Context.Version) return null;
                    if (t.Target() > Context.Target) return null;
                    return (Type)t;
                }).Where(t => t != null).ToReadOnly();

                return all.Except(Opaque).ToReadOnly();
            }
        }

        public static ReadOnlyCollection<ClrType> ScalarClr
        {
            get { return Scalar.Select(t => t.ClrType).ToReadOnly(); }
        }

        public static ReadOnlyCollection<Type> Vector
        {
            get { return Scalar.SelectMany(t => new[] {t.v1, t.v2, t.v4}).Where(t => t.SizeOfElement <= 128 / 8).ToReadOnly(); }
        }

        public static ReadOnlyCollection<ClrType> VectorClr
        {
            get { return Vector.Select(t => t.ClrType).ToReadOnly(); }
        }
    }
}
