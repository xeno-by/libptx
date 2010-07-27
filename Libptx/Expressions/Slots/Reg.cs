using System;
using System.IO;
using Libptx.Common;
using Type=Libptx.Common.Types.Type;
using XenoGears.Assertions;

namespace Libptx.Expressions.Slots
{
    public partial class Reg : Atom, Slot, Expression
    {
        public String Name { get; set; }
        public Type Type { get; set; }

        private int _alignment;
        public int Alignment
        {
            get
            {
                if (_alignment == 0)
                {
                    if (this.is_pred()) return 1;
                    if (this.is_opaque()) return 16;
                    return this.SizeOfElement();
                }
                else
                {
                    return _alignment;
                }
            }

            set { _alignment = value; }
        }

        protected override void CustomValidate(Module ctx)
        {
            (Type != null).AssertTrue();
            Type.Validate(ctx);
            this.is_opaque().AssertFalse();
            this.is_ptr().AssertFalse();

            if (Name != null) Name.ValidateName();
            if (Alignment != 0) Alignment.ValidateAlignment(Type);
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}