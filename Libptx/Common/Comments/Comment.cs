using System;
using System.Diagnostics;
using System.IO;
using Libptx.Expressions;
using Libptx.Statements;
using XenoGears.Functional;
using XenoGears.Strings.Writers;

namespace Libptx.Common.Comments
{
    [DebuggerNonUserCode]
    public class Comment : Atom, Statement
    {
        private String _text = String.Empty;
        public String Text
        {
            get { return _text; }
            set { _text = value ?? String.Empty; }
        }

        public static implicit operator Comment(String text)
        {
            return text == null ? null : new Comment { Text = text };
        }

        protected override void RenderPtx()
        {
            if (Text.IsEmpty()) return;

            if (Text.Trim().IsEmpty())
            {
                var indented = writer.InnerWriter as IndentedWriter;
                if (Text == Environment.NewLine && indented != null)
                {
                    indented.WriteNoTabs(Text);
                }
                else
                {
                    writer.Write(Text);
                }
            }
            else
            {
                var inline = ctx.Parent is Expression && !(ctx.Parent is Label);
                if (inline) writer.Write("/* {0} */ ", Text);
                else writer.Write("// {0}", Text);
            }
        }
    }
}