using System;
using System.Diagnostics;
using System.IO;
using Libptx.Statements;
using XenoGears.Functional;
using XenoGears.Strings;

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

        // todo. this is a dirty hack - get rid of it
        public static bool Inline { get; set; }

        protected override void RenderAsPtx(TextWriter writer)
        {
            if (Text.IsEmpty()) return;

            if (Text.Trim().IsEmpty())
            {
                var indented = writer as IndentedTextWriter;
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
                if (Inline) writer.Write("/* {0} */ ", Text);
                else writer.WriteLine("// {0}", Text);
            }
        }
    }
}