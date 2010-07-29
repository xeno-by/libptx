using System.Diagnostics;
using System.IO;
using Libptx.Common;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions.Immediate
{
    [DebuggerNonUserCode]
    public class WarpSz : Atom, Expression
    {
        public Type Type
        {
            get { return typeof(uint); }
        }

        protected override void RenderPtx()
        {
            writer.Write("WARP_SZ");
        }
    }
}