using System;
using Libptx.Common.Types;
using Libptx.Expressions;
using ClrType=System.Type;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Edsl.Vars
{
    public class var : Var
    {
        public static bool operator ==(Type t, var v) { throw new NotImplementedException(); }
        public static bool operator !=(Type t, var v) { return !(t == v); }
        public static bool operator ==(var v, Type t) { return t == v; }
        public static bool operator !=(var v, Type t) { return t != v; }

        public static bool operator ==(TypeName t, var v) { throw new NotImplementedException(); }
        public static bool operator !=(TypeName t, var v) { return !(t == v); }
        public static bool operator ==(var v, TypeName t) { return t == v; }
        public static bool operator !=(var v, TypeName t) { return t != v; }

        public static bool operator ==(ClrType t, var v) { throw new NotImplementedException(); }
        public static bool operator !=(ClrType t, var v) { return !(t == v); }
        public static bool operator ==(var v, ClrType t) { return t == v; }
        public static bool operator !=(var v, ClrType t) { return t != v; }

        public override bool Equals(Object obj) { return base.Equals(obj); }
        public override int GetHashCode() { return base.GetHashCode(); }
    }
}
