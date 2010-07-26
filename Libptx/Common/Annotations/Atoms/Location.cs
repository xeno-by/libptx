using System;
using System.Diagnostics;

namespace Libptx.Common.Annotations.Atoms
{
    [DebuggerNonUserCode]
    public class Location
    {
        public String File { get; set; }
        public int Line { get; set; }
        public int Column { get; set; }
    }
}