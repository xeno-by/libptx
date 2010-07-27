using System;
using System.Diagnostics;
using System.IO;
using Libptx.Common;
using XenoGears.Assertions;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions.Sregs
{
    [DebuggerNonUserCode]
    public abstract partial class Sreg : Atom, Expression
    {
        public Type Type
        {
            get { throw new NotImplementedException(); }
        }

        protected override void CustomValidate(Module ctx)
        {
            (Type != null).AssertTrue();
            Type.Validate(ctx);
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}