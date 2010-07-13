using System;

namespace Libptx.Common.Infrastructure
{
    public class Location
    {
        public virtual String File { get; set; }
        public virtual int Line { get; set; }
        public virtual int Column { get; set; }
    }
}