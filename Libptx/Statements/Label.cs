using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Libptx.Common;
using Libptx.Common.Types.Pointers;
using Libptx.Expressions;
using Libptx.Expressions.Addresses;
using XenoGears.Assertions;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Statements
{
    [DebuggerNonUserCode]
    public class Label : Atom, Expression, Statement, Addressable
    {
        public String Name { get; set; } // may be null
        public Type Type { get { return typeof(Ptr); } }

        protected override void CustomValidate(Module ctx)
        {
            // todo. what else do we validate here?

            Func<Block, int> cntInBlock = null;
            cntInBlock = blk => blk.Stmts.Count(s =>
            {
                var lbl = s as Label;
                return lbl != null && lbl.Name == this.Name && lbl.Name != null;
            }) + blk.Stmts.OfType<Block>().Sum(sub => cntInBlock(sub));

            ctx.Entries.AssertNone(f => cntInBlock(f.Body) > 1);
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}