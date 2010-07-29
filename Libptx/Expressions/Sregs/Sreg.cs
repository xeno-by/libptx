using System.Diagnostics;
using System.IO;
using Libptx.Common;
using Libptx.Reflection;
using Type = Libptx.Common.Types.Type;
using XenoGears.Assertions;

namespace Libptx.Expressions.Sregs
{
    [DebuggerNonUserCode]
    public abstract partial class Sreg : Atom, Expression
    {
        public Type Type
        {
            get { return this.SregSig().AssertNotNull().Type; }
        }

        protected override void CustomValidate(Module ctx)
        {
            (Type != null).AssertTrue();
            Type.Validate(ctx);
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            writer.Write(this.Sig());
        }
    }
}