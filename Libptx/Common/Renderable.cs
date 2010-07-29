using System;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace Libptx.Common
{
    public interface Renderable
    {
        void RenderAsPtx(TextWriter writer);
    }

    [DebuggerNonUserCode]
    public static class RenderableExtensions
    {
        public static String RenderAsPtx(this Renderable renderable)
        {
            if (renderable == null)
            {
                return null;
            }
            else
            {
                var buf = new StringBuilder();
                renderable.RenderAsPtx(new StringWriter(buf));
                return buf.ToString();
            }
        }

        public static void RenderAsPtx(this Renderable renderable, TextWriter writer)
        {
            if (renderable == null) return;
            ((Renderable)renderable).RenderAsPtx(writer);
        }
    }
}