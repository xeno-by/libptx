using System;
using System.IO;
using Libptx.Common;
using Libptx.Common.Enumerations;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions
{
    public class Var : Atom, Expression, Addressable
    {
        public String Name { get; set; } // may be null
        public Space Space { get; set; }
        public Type Type { get; set; } // must not be null
        public Const Init { get; set; }
        public int Alignment { get; set; } // non-negative
        public bool IsVisible { get; set; }
        public bool IsExtern { get; set; }
        public VarMod Mod { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            throw new NotImplementedException();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}
