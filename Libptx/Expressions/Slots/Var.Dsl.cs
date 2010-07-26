using Libptx.Common.Enumerations;
using Libptx.Expressions.Addresses;
using XenoGears.Assertions;

namespace Libptx.Expressions.Slots
{
    public partial class Var
    {
        public Index this[int offset]
        {
            get
            {
                (Space != space.reg).AssertTrue();
                return new Index { Base = this, Offset = offset };
            }
        }

        public static Address operator +(Var @var, int offset)
        {
            if (@var == null) return null;
            if (@var.Space == space.reg) return new Address{Base = null, Offset = new Offset{Base = @var, Imm = offset}};
            return new Address{Base = @var, Offset = new Offset{Base = null, Imm = offset}};
        }

        public static Address operator +(Var @var, long offset)
        {
            if (@var == null) return null;
            if (@var.Space == space.reg) return new Address{Base = null, Offset = new Offset{Base = @var, Imm = offset}};
            return new Address{Base = @var, Offset = new Offset{Base = null, Imm = offset}};
        }

        public static Address operator -(Var @var, int offset)
        {
            return @var + (-offset);
        }

        public static Address operator -(Var @var, long offset)
        {
            return @var + (-offset);
        }
    }
}
