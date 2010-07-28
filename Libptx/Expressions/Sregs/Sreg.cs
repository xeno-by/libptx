using System;
using System.Diagnostics;
using System.IO;
using Libptx.Common;
using Libptx.Expressions.Sregs.Annotations;
using XenoGears.Assertions;
using XenoGears.Reflection.Attributes;
using Type=Libptx.Common.Types.Type;
using Libptx.Reflection;

namespace Libptx.Expressions.Sregs
{
    [DebuggerNonUserCode]
    public abstract partial class Sreg : Atom, Expression
    {
        public Type Type
        {
            get { return this.GetType().Attr<SregAttribute>().Type; }
        }

        protected override void CustomValidate(Module ctx)
        {
            (Type != null).AssertTrue();
            Type.Validate(ctx);
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            writer.Write(this.Signature());
        }
    }
}