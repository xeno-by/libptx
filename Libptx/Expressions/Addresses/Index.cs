using System;
using System.Diagnostics;
using System.IO;
using Libptx.Common;
using Libptx.Common.Types.Pointers;
using Libptx.Expressions.Slots;
using Type=Libptx.Common.Types.Type;
using XenoGears.Assertions;

namespace Libptx.Expressions.Addresses
{
    [DebuggerNonUserCode]
    public class Index : Atom, Expression
    {
        public Var Base { get; set; }
        public Offset Offset { get; set; }

        public Type Type
        {
            get { return typeof(Ptr); }
        }

        protected override void CustomValidate(Module ctx)
        {
            (Base != null && Offset != null).AssertTrue();
            (Base.is_arr()).AssertTrue();
            Base.Validate(ctx);
            Offset.Validate(ctx);
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}