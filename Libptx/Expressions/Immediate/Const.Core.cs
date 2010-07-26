using System;
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
            throw new NotImplementedException();
        }
    }
}