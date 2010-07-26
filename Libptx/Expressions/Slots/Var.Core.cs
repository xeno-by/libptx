using System;
using System.Diagnostics;
using System.IO;
using Libptx.Common;
using Libptx.Common.Enumerations;
using Libptx.Expressions.Addresses;
using Libptx.Expressions.Immediate;
using Type = Libptx.Common.Types.Type;

namespace Libptx.Expressions.Slots
{
    [DebuggerNonUserCode]
    public partial class Var : Atom, Slot, Addressable
    {
        public String Name { get; set; } // may be null
        public space Space { get; set; }
        public Type Type { get; set; } // must not be null

        public Const Init { get; set; }
        public int Alignment { get; set; } // non-negative, multiple of Type element's size
        public bool IsVisible { get; set; }
        public bool IsExtern { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            // todo. arrays and regs
            // todo. Initializers are allowed for all types except .f16 and .pred.
            // todo. Currently, variable initialization is supported only for constant and global state spaces
            // todo. declared must not have types: sreg, ptr
            // todo. preds must be reg
            // todo. other opaques must be global
            // todo. A texture base address is assumed to be aligned to a 16-byte address?!
            // todo. A surface base address is assumed to be aligned to a 16-byte address?!
            throw new NotImplementedException();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}
