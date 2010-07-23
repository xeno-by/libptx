﻿using System;
using System.IO;
using Libptx.Common.Enumerations;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions
{
    public abstract class Var_Aux : Expression
    {
        protected abstract Type GetExpressionType();
        public sealed override Type Type { get { return GetExpressionType(); } }
    }

    public partial class Var : Var_Aux, Addressable
    {
        public Var Base { get; set; }
        public VarMod Mod { get; set; }

        public String Name { get; set; } // may be null
        public space Space { get; set; }
        public new Type Type { get; set; } // must not be null
        protected override Type GetExpressionType() { return Type; }
        public Const Init { get; set; }
        public int Alignment { get; set; } // non-negative, multiple of Type element's size
        public bool IsVisible { get; set; }
        public bool IsExtern { get; set; }

        public Var() { Space = space.reg; }
        public int SizeInMemory { get { return Type.SizeInMemory; } }
        public int SizeOfElement { get { return Type.SizeOfElement; } }

        protected override void CustomValidate(Module ctx)
        {
            // todo. arrays and regs
            // todo. Initializers are allowed for all types except .f16 and .pred.
            // todo. Currently, variable initialization is supported only for constant and global state spaces
            throw new NotImplementedException();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}
