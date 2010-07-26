using System;
using System.Diagnostics;
using System.IO;
using Libptx.Common;
using Libptx.Common.Spaces;
using Libptx.Expressions.Addresses;
using Libptx.Expressions.Immediate;
using Type = Libptx.Common.Types.Type;
using XenoGears.Assertions;

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
            this.is_opaque().AssertImplies(Space == param || Space == global);
            (Init != null).AssertImplies(Space.is_const() || Space == global);

            // todo. arrays and regs
            // todo. Initializers are allowed for all types except .f16 and .pred.
            // todo. declared must not have types: sreg, ptr
            // todo. preds must be reg
            // todo. other opaques must be global/param
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
