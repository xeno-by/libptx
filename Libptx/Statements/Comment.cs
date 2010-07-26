using System;
using System.IO;

namespace Libptx.Statements
{
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