using System.Diagnostics;
using Libptx.Common.Contexts;

namespace Libptx.Common
{
    public interface Validatable
    {
        void Validate();
    }

    [DebuggerNonUserCode]
    public static class ValidatableExtensions
    {
        public static void Validate(this Validatable validatable)
        {
            var ctx = ValidationContext.Current ?? new ValidationContext(null);
            Validate(validatable, ctx);
        }

        public static void Validate(this Validatable validatable, Module ctx)
        {
            var curr = ValidationContext.Current;
            if (curr != null && curr.Module == ctx)
            {
                Validate(validatable);
            }
            else
            {
                Validate(validatable, new ValidationContext(ctx));
            }
        }

        public static void Validate(this Validatable validatable, ValidationContext ctx)
        {
            if (validatable == null) return;
            if (ctx == null) Validate(validatable);

            using (ValidationContext.Push(ctx))
            {
                validatable.Validate();
            }
        }
    }
}