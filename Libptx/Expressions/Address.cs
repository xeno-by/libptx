namespace Libptx.Expressions
{
    public class Address
    {
        public Addressable Base { get; set; } // may be null
        public Var Offset1 { get; set; } // may be null
        public int Offset2 { get; set; } // also support uint and longs
    }
}