using System;
using System.IO;
using System.Linq;
using Libptx.Common;
using Libptx.Expressions;
using XenoGears.Assertions;

namespace Libptx.Statements
{
    public class Label : Atom, Expression, Statement
    {
        public String Name { get; set; } // may be null

        protected override void CustomValidate(Module ctx)
        {
            Func<Block, int> cntInBlock = null;
            cntInBlock = blk => blk.Stmts.Count(s => s == this) + blk.Stmts.OfType<Block>().Sum(sub => cntInBlock(sub));

            ctx.Funcs.AssertNone(f => cntInBlock(f.Body) > 1);
            ctx.Entries.AssertNone(f => cntInBlock(f.Body) > 1);
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}