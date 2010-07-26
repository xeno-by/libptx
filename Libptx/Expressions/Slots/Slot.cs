using System;
using Libptx.Common.Enumerations;

namespace Libptx.Expressions.Slots
{
    public interface Slot : Expression
    {
        String Name { get; }
        space Space { get; }
    }

    public static class SlotExtensions
    {
        public static int SizeInMemory(this Slot var)
        {
            if (var == null) return 0;
            if (var.Type == null) return 0;
            return var.Type.SizeInMemory;
        }

        public static int SizeOfElement(this Slot var)
        {
            if (var == null) return 0;
            if (var.Type == null) return 0;
            return var.Type.SizeOfElement;
        }
    }
}