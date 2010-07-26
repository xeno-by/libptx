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
        public Const(Object value)
        {
            Value = value;
        }

        private Object _value;
        public Object Value
        {
            get { return _value; }
            set
            {
                ValidateValue(value);
                _value = value;
            }
        }

        public Type Type
        {
            get { return _value == null ? null : _value.GetType(); }
        }

        private void ValidateValue(Object value)
        {
            value.AssertNotNull();

            if (value is Addressable) return;

            var elt = value.GetType().Unfold(t => t.IsArray ? t.GetElementType() : null, t => t != null).Last();
            (elt.IsCudaPrimitive() || elt.IsCudaVector()).AssertTrue();
        }

        protected override void CustomValidate(Module ctx)
        {
            ValidateValue(Value);
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