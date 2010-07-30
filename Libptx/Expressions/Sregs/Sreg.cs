using System;
using System.Diagnostics;
using Libcuda.Versions;
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

        protected override SoftwareIsa CustomVersion { get { return this.SregSig().Version; } }
        protected override HardwareIsa CustomTarget { get { return this.SregSig().Target; } }

        protected override void CustomValidate()
        {
            (Type != null).AssertTrue();
            Type.Validate();
        }

        protected override void RenderPtx()
        {
            writer.Write(this.Signature());
        }
    }
}