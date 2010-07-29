using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using Libptx.Common;
using Type = Libptx.Common.Types.Type;

namespace Libptx.Expressions.Slots
{
    // I wish, we had mixins
    // so that I don't have to duplicate slot-relared logic in Var and Reg

    public interface Slot : Validatable, Renderable
    {
        String Name { get; set; }
        Type Type { get; set; }
        int Alignment { get; set; }

        void RenderDeclarationAsPtx(TextWriter writer);
    }

    [DebuggerNonUserCode]
    public static class SlotExtensions
    {
        public static int SizeInMemory(this Slot slot)
        {
            if (slot == null) return 0;
            if (slot.Type == null) return 0;
            return slot.Type.SizeInMemory;
        }

        public static int SizeOfElement(this Slot slot)
        {
            if (slot == null) return 0;
            if (slot.Type == null) return 0;
            return slot.Type.SizeOfElement;
        }

        public static String RenderDeclarationAsPtx(this Slot slot)
        {
            if (slot == null)
            {
                return null;
            }
            else
            {
                var buf = new StringBuilder();
                slot.RenderDeclarationAsPtx(new StringWriter(buf));
                return buf.ToString();
            }
        }

        public static void RenderDeclarationAsPtx(this Slot slot, TextWriter writer)
        {
            if (slot == null) return;
            ((Slot)slot).RenderDeclarationAsPtx(writer);
        }
    }
}