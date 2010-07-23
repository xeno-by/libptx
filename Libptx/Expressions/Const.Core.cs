using System;
using System.IO;
using Libcuda.DataTypes;
using Libptx.Common;
using Libptx.Statements;
using XenoGears.Assertions;
using XenoGears.Functional;
using System.Linq;

namespace Libptx.Expressions
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

        private void ValidateValue(Object value)
        {
            value.AssertNotNull();

            if (value is Var) return;
            if (value is Label) return;

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