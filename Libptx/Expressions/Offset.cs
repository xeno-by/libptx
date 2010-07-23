namespace Libptx.Expressions
{
    public class Offset
    {
        public Var Var { get; set; }
        public long Imm { get; set; }

        public static implicit operator Offset(Var @var)
        {
            return @var == null ? null : new Offset { Var = @var };
        }

        public static implicit operator Offset(long offset)
        {
            return new Offset { Imm = offset };
        }
    }
}