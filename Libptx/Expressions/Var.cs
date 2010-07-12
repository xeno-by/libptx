using System;

namespace Libptx.Expressions
{
    public class Var : Addressable
    {
        public String Name { get; set; } // may be null
        public Space Space { get; set; }
        public Type Type { get; set; } // must not be null
        public Const Init { get; set; }
        public int Alignment { get; set; } // non-negative
        public bool IsVisible { get; set; }
        public bool IsExtern { get; set; }
        public VarMod Mod { get; set; }
    }
}
