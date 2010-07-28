using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using Libptx.Common.Types;
using XenoGears.Functional;
using Type = Libptx.Common.Types.Type;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public static class Types
    {
        private static readonly ReadOnlyCollection<Type> _scalar;
        private static readonly ReadOnlyCollection<Type> _vector;
        private static readonly ReadOnlyCollection<Type> _opaque;
        private static readonly ReadOnlyCollection<Type> _other;

        static Types()
        {
            _opaque = new Type[]{TypeName.Texref, TypeName.Samplerref, TypeName.Surfref}.ToReadOnly();
            _other = new Type[]{TypeName.Pred, TypeName.Ptr, TypeName.Bmk}.ToReadOnly();
            _scalar = Enum.GetValues(typeof(TypeName)).Cast<TypeName>().Select(t => (Type)t).Except(_opaque, _other).ToReadOnly();
            _vector = _scalar.SelectMany(t => new[] { t.v1, t.v2, t.v4 }).Where(t => t.SizeOfElement <= 128 / 8).ToReadOnly();
        }

        public static ReadOnlyCollection<Type> Scalar
        {
            get { return _scalar; }
        }

        public static ReadOnlyCollection<Type> Vector
        {
            get { return _vector; }
        }

        public static ReadOnlyCollection<Type> Opaque
        {
            get { return _opaque; }
        }

        public static ReadOnlyCollection<Type> Other
        {
            get { return _other; }
        }
    }
}