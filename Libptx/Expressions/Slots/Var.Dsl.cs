using Libptx.Common.Spaces;
using Libptx.Expressions.Addresses;
using XenoGears.Assertions;

namespace Libptx.Expressions.Slots
{
    public partial class Var
    {
        // impure, but convenient
        // now we can write like that: "arr[foo + 2]"

        public Index this[Address address]
        {
            get
            {
                // this is the only place (or one of the few places) where we perform early validation
                // however, here it's justified, since silently ignoring non-empty address.Base might lead to mysterious behavior
                // whereas stupid mistakes like forgetting to provide a mandatory attribute of an instruction will crash anyways
                (address != null && address.Base == null).AssertTrue();
                return new Index{Base = this, Offset = address.Offset};
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
