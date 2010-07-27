using System;
using System.Diagnostics;
using System.IO;
using XenoGears.Assertions;

namespace Libptx.Common.Debug
{
    [DebuggerNonUserCode]
    public class Location : Atom
    {
        public String File { get; set; }
        public int Line { get; set; }
        public int Column { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            File.AssertNotNull();
            (Line >= 0).AssertTrue();
            (Column >= 0).AssertTrue();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}