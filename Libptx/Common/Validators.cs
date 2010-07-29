using System;
using System.Diagnostics;
using Libptx.Reflection;
using XenoGears.Assertions;
using XenoGears.Functional;
using Type = Libptx.Common.Types.Type;
using XenoGears.Strings;
using System.Linq;

namespace Libptx.Common
{
    [DebuggerNonUserCode]
    public static class Validators
    {
        public static void ValidateName(this String name)
        {
            name.AssertNotNull();

            var fmt1 = name.Match("^[a-zA-Z][a-zA-Z0-9_$]*$");
            var fmt2 = name.Match("^[_$%][a-zA-Z0-9_$]*$");
            (fmt1.Success || fmt2.Success).AssertTrue();

            var sregs = Sregs.Sigs.Select(sig => sig.Name).ToHashSet();
            sregs.Contains(name).AssertFalse();
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