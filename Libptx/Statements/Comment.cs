using System;
using System.Diagnostics;
using System.IO;

namespace Libptx.Statements
{
    [DebuggerNonUserCode]
    public class Comment : Statement
    {
        public String Text { get; set; }

        public void Validate(Module ctx)
        {
            throw new NotImplementedException();
        }

        public void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}