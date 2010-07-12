using System;
using Libptx.Expressions;

namespace Libptx.Statements
{
    public class Label : Expression, Statement
    {
        public String Name { get; set; } // may be null
    }
}