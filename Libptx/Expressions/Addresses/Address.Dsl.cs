using Libptx.Expressions.Addresses;
using Libptx.Expressions.Slots;

namespace Libptx.Expressions.Addresses
{
    public partial class Address
    {
        public static implicit operator Address(int offset)
        {
            return new Address { Offset = offset };
        }

        public static implicit operator Address(long offset)
        {
            return new Address { Offset = offset };
        }

        public static implicit operator Address(Offset offset)
        {
            return offset == null ? null : new Address { Offset = offset };
        }

        public static implicit operator Address(Reg reg)
        {
            return reg == null ? null : new Address { Offset = reg };
        }

        public static implicit operator Address(Var @var)
        {
            return @var == null ? null : new Address { Offset = @var };
        }
    }

    public partial class Offset
    {
        public static implicit operator Offset(int offset)
        {
            return new Offset { Imm = offset };
        }

        public static implicit operator Offset(long offset)
        {
            return new Offset { Imm = offset };
        }

        public static implicit operator Offset(Reg reg)
        {
            return reg == null ? null : new Offset { Base = reg };
        }

        public static implicit operator Offset(Var @var)
        {
            return @var == null ? null : new Offset { Base = @var };
        }
    }
}

namespace Libptx.Expressions.Slots
{
    public partial class Reg
    {
        public static Offset operator +(Reg reg, int offset)
        {
            if (reg == null) return null;
            return new Offset { Base = reg, Imm = offset };
        }

        public static Offset operator +(Reg reg, long offset)
        {
            if (reg == null) return null;
            return new Offset { Base = reg, Imm = offset };
        }

        public static Offset operator -(Reg reg, int offset)
        {
            return reg + (-offset);
        }

        public static Offset operator -(Reg reg, long offset)
        {
            return reg + (-offset);
        }
    }

    public partial class Var
    {
        public Address this[Offset offset]
        {
            get { return new Address { Base = this, Offset = offset }; }
        }

        public static Offset operator +(Var @var, int offset)
        {
            if (@var == null) return null;
            return new Offset { Base = @var, Imm = offset };
        }

        public static Offset operator +(Var @var, long offset)
        {
            if (@var == null) return null;
            return new Offset { Base = @var, Imm = offset };
        }

        public static Offset operator -(Var @var, int offset)
        {
            return @var + (-offset);
        }

        public static Offset operator -(Var @var, long offset)
        {
            return @var + (-offset);
        }
    }
}
