using System;
using System.Diagnostics;
using System.IO;
using Libptx.Common;

namespace Libptx.Statements
{
    [DebuggerNonUserCode]
    public class Comment : Atom, Statement
    {
        public String Text { get; set; }

        protected override void RenderAsPtx(TextWriter writer)
        {
            // todo. smartly choose between /**/ and //
            throw new NotImplementedException();
        }
    }
}