using System;
using Libptx.Common;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions
{
    public abstract class Expression : Atom
    {
        public abstract Type Type { get; }

        // impure, tho convenient
        public static bool operator !=(Type type, Expression expr) { return expr != type; }
        public static bool operator ==(Type type, Expression expr) { return expr == type; }
        public static bool operator !=(Expression expr, Type type) { return !(expr == type); }
        public static bool operator ==(Expression expr, Type type)
        {
            var expr_is_null = ReferenceEquals(expr, null);
            var type_is_null = ReferenceEquals(type, null);
            if (expr_is_null || type_is_null) return expr_is_null && type_is_null;
            return expr.Type == type;
        }

        // so that the compiler doesn't whine
        public override bool Equals(Object obj) { return base.Equals(obj); }
        public override int GetHashCode() { return base.GetHashCode(); }
    }
}