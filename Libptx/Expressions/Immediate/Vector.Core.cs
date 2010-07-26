using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using Libptx.Common;
using Libptx.Common.Types;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Expressions.Immediate
{
    public partial class Vector : Atom, Expression
    {
        private Type _elementType;
        public Type ElementType
        {
            get
            {
                if (_elementType != null) return _elementType;
                return Elements.Select(el => el == null ? null : el.Type).FirstOrDefault();
            }
            set
            {
                _elementType = value;
            }
        }

        private IList<Expression> _elements = new List<Expression>();
        public IList<Expression> Elements
        {
            get { return _elements; }
            set { _elements = value ?? new List<Expression>(); }
        }

        public Type Type
        {
            get
            {
                var vec_type = ElementType;
                if (vec_type != null && Elements.Count() == 1) vec_type.Mod |= TypeMod.V1;
                if (vec_type != null && Elements.Count() == 2) vec_type.Mod |= TypeMod.V2;
                if (vec_type != null && Elements.Count() == 3) vec_type.Mod |= TypeMod.V4;
                if (vec_type != null && Elements.Count() == 4) vec_type.Mod |= TypeMod.V4;
                return vec_type;
            }
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