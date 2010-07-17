using System;
using System.IO;
using System.Text;

namespace Libptx.Common
{
    public static class Extensions
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
    }
}