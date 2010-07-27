using System;
using System.Diagnostics;
using System.IO;
using Libcuda.DataTypes;
using Libptx.Common;
using Libptx.Expressions.Addresses;
using XenoGears.Assertions;
using XenoGears.Functional;
using System.Linq;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions.Immediate
{
    [DebuggerNonUserCode]
    public partial class Const : Atom, Expression
    {
        public Const() {}
        public Const(Object value) { Value = value; }

        public Object Value { get; set; }
        public Type Type { get { return Value == null ? null : Value.GetType(); } }

        protected override void CustomValidate(Module ctx)
        {
            (Type != null).AssertTrue();
            Type.Validate(ctx);

            (Value != null).AssertTrue();
            if (Value is Addressable) return;

            var elt = Value.GetType().Unfold(t => t.IsArray ? t.GetElementType() : null, t => t != null).Last();
            (elt.IsCudaPrimitive() || elt.IsCudaVector()).AssertTrue();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            // todo.
            // 1) true/false -> 1/0
            // 2) 0x... with optional U (mandatory uppercase)
            // 3) getbits for f (0f... and 0d....)
            // 4) also support opaques
            throw new NotImplementedException();
        }
    }
}