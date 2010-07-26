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
            (@var.Space != space.reg).AssertTrue();
            return new Address { Base = @var, Offset = offset };
        }

        public static Address operator +(Var @var, long offset)
        {
            if (@var == null) return null;
            (@var.Space != space.reg).AssertTrue();
            return new Address { Base = @var, Offset = offset };
        }

        public static Address operator -(Var @var, int offset)
        {
            if (@var == null) return null;
            (@var.Space != space.reg).AssertTrue();
            return new Address { Base = @var, Offset = -offset };
        }

        public static Address operator -(Var @var, long offset)
        {
            if (@var == null) return null;
            (@var.Space != space.reg).AssertTrue();
            return new Address { Base = @var, Offset = -offset };
        }
    }
}
