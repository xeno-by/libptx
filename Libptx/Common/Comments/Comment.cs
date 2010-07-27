using System;
using System.Diagnostics;
using System.IO;
using Libptx.Statements;

namespace Libptx.Common.Comments
{
    [DebuggerNonUserCode]
    public class Comment : Atom, Statement
    {
        public String Text { get; set; }

        public static implicit operator Comment(String text)
        {
            return text == null ? null : new Comment { Text = text };
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            // todo. smartly choose between /**/ and //
            throw new NotImplementedException();
        }
    }
}