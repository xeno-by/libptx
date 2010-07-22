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
        public space Space { get; set; }
        public Type Type { get; set; } // must not be null
        public int Size { get { return Type.Size; } }
        public Const Init { get; set; }
        public int Alignment { get; set; } // non-negative
        public VarMod Mod { get; set; }
        public bool IsVisible { get; set; }
        public bool IsExtern { get; set; }

        // todo. inits are only allowed for <what> spaces?

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
