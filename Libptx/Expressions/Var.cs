using System;
using Libptx.Common.Enumerations;
using Libptx.Common.Infrastructure;
using Type=Libptx.Common.Type;

namespace Libptx.Expressions
{
    public class Var : Atom, Addressable
    {
        public virtual String Name { get; set; } // may be null
        public virtual Space Space { get; set; }
        public virtual Type Type { get; set; } // must not be null
        public virtual Const Init { get; set; }
        public virtual int Alignment { get; set; } // non-negative
        public virtual bool IsVisible { get; set; }
        public virtual bool IsExtern { get; set; }
        public virtual VarMod Mod { get; set; }

        public override void Validate()
        {
            throw new NotImplementedException();
        }
    }
}
