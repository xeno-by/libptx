using System;
using System.IO;
using Libptx.Common;
using Libptx.Common.Enumerations;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions.Slots
{
    public abstract partial class Special : Atom, Slot
    {
        public String Name
        {
            get { throw new NotImplementedException(); }
        }

        public space Space
        {
            get { return space.sreg; }
        }

        public Type Type
        {
            get { throw new NotImplementedException(); }
        }

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