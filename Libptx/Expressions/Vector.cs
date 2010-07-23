using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions
{
    public class Vector : Expression
    {
        public Type ElementType { get; set; }

        private IList<Var> _elements = new List<Var>();
        public IList<Var> Elements
        {
            get { return _elements; }
            set { _elements = value ?? new List<Var>(); }
        }

        public override Type Type
        {
            get
            {
                if (ElementType != null) return ElementType;
                return Elements.Select(el => el == null ? null : el.Type).FirstOrDefault();
            }
        }

        public static implicit operator Vector(Var[] vars)
        {
            throw new NotImplementedException();
        }

        protected override void CustomValidate(Module ctx)
        {
            throw new NotImplementedException();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}