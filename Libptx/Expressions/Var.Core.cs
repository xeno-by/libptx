using System;
using System.IO;
using Libptx.Common;
using Libptx.Common.Enumerations;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions
{
    public partial class Var : Atom, Expression, Addressable
    {
        public Var Base { get; set; }
        public VarMod Mod { get; set; }

        public String Name { get; set; } // may be null
        public space Space { get; set; }
        public Type Type { get; set; } // must not be null
        public Const Init { get; set; }
        public int Alignment { get; set; } // non-negative, multiple of Type element's size
        public bool IsVisible { get; set; }
        public bool IsExtern { get; set; }

        public Var() { Space = space.reg; }
        public int Size { get { return Type.Size; } }

        protected override void CustomValidate(Module ctx)
        {
            // todo.Initializers are allowed for all types except .f16 and .pred.
            // todo. inits are only allowed for <what> spaces?
            throw new NotImplementedException();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}
