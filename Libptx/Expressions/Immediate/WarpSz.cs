using System.IO;
using Libptx.Common;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions.Immediate
{
    public class WarpSz : Atom, Expression
    {
        public Type Type
        {
            get { return typeof(uint); }
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            writer.Write("WARP_SZ");
        }
    }
}