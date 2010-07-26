using System;
using System.Diagnostics;
using System.IO;
using Libptx.Common;
using Libptx.Common.Spaces;
using Libptx.Expressions.Addresses;
using Libptx.Expressions.Immediate;
using Type = Libptx.Common.Types.Type;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Expressions.Slots
{
    [DebuggerNonUserCode]
    public partial class Var : Atom, Slot, Addressable
    {
        public String Name { get; set; }
        public space Space { get; set; }
        public Type Type { get; set; }

        public Const Init { get; set; }

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

        public bool IsVisible { get; set; }
        public bool IsExtern { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            (Space != 0).AssertTrue();
            (Space != sreg).AssertTrue();

            (Type != null).AssertTrue();
            Type.Validate(ctx);
            this.is_pred().AssertImplies(Space == reg);
            this.is_ptr().AssertFalse();
            this.is_opaque().AssertImplies(Space == param || Space == global);
            this.vec_rank().AssertThat(rank => rank == 0 || rank == 2 || rank == 4);
            this.is_arr().AssertImplies(Space != reg);

            agree_or_null(Init, Type);
            (Init != null).AssertImplies(Space.is_const() || Space == global);
            (Init != null).AssertImplies(Type != f16 && Type != pred);
            // todo. ".global variables used in initializers, the resulting address is a generic address"
            // wtf does this mean? is this feature limited to SM_20 or that was just unintended pun?
            var init_var = Init == null ? null : Init.Value as Var;
            if (init_var != null) (init_var.Space.is_const() || init_var.Space == global).AssertTrue();

            (Alignment > 0).AssertTrue();
            (_alignment != 0).AssertImplies(!this.is_pred() && !this.is_opaque());
            Alignment.Unfold(i => i / 2, i => i != 1).AssertEach(i => i % 2 == 0);
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}
