using System;

namespace Libptx.Common.Annotations.Atoms
{
    public class Location
    {
        public String File { get; set; }
        public int Line { get; set; }
        public int Column { get; set; }
    }
}