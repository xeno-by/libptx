namespace Libptx.Expressions
{
    public partial class Var
    {
        public Address this[int offset]
        {
            get { return new Address { Base = this, Offset = offset * Type.Size }; }
        }

        public static Address operator +(Var @var, int offset)
        {
            return @var == null ? null : new Address { Base = @var, Offset = offset };
        }

        public static Address operator -(Var @var, int offset)
        {
            return @var == null ? null : new Address { Base = @var, Offset = -offset };
        }
    }
}
