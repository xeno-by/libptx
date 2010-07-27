using System;
using XenoGears.Assertions;
using XenoGears.Functional;
using Type = Libptx.Common.Types.Type;

namespace Libptx.Common
{
    public static class Validators
    {
        public static void ValidateName(this String name)
        {
            // todo. name cannot be null
            // todo. also see section 4.4 of PTX ISA spec
            throw new NotImplementedException();
        }

        public static void ValidateAlignment(this int alignment, Type type)
        {
            (alignment > 0).AssertTrue();
            alignment.Unfold(i => i / 2, i => i != 1).AssertEach(i => i % 2 == 0);

            type.AssertNotNull();
            if (type.SizeOfElement != 0) (alignment % type.SizeOfElement == 0).AssertTrue();
        }
    }
}