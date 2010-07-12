using Libptx.Common.Infrastructure;

namespace Libptx.Expressions
{
    public class Address : Atom
    {
        public virtual Addressable Base { get; set; } // may be null
        public virtual Var Offset1 { get; set; } // may be null
        public virtual int Offset2 { get; set; } // also support uint and longs
    }
}