using System;
using ClrType = System.Type;

namespace Libptx.Expressions
{
    public class Type
    {
        public TypeName Name { get; set; }
        public TypeMod Mod { get; set; }
        public int[] Dims { get; set; }

        public static bool operator ==(Type t1, Type t2) { throw new NotImplementedException(); }
        public static bool operator !=(Type t1, Type t2) { return !(t1 == t2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }
        public override String ToString() { throw new NotImplementedException(); }

        public static implicit operator Type(ClrType type) { throw new NotImplementedException(); }
        public static implicit operator ClrType(Type type) { throw new NotImplementedException(); }

        public static implicit operator String(Type type) { throw new NotImplementedException(); }
        public static implicit operator Type(String type) { throw new NotImplementedException(); }
    }
}