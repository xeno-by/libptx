using Libptx.Common.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Expressions
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

        public static Address operator -(Var @var, int offset)
        {
            if (@var == null) return null;
            (@var.Space != space.reg).AssertTrue();
            return new Address { Base = @var, Offset = -offset };
        }
    }
}
