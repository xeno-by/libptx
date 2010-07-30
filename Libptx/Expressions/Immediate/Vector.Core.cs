using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using System.Collections.Generic;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Common.Types;
using Libptx.Expressions.Slots;
using XenoGears.Assertions;
using Type=Libptx.Common.Types.Type;
using XenoGears.Functional;
using Libptx.Reflection;

namespace Libptx.Expressions.Immediate
{
    [DebuggerNonUserCode]
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

                if (Elements.Count() == 1)
                {
                    if (vec_type != null) vec_type.Mod |= TypeMod.V1;
                }
                else if (Elements.Count() == 2) 
                {
                    if (vec_type != null) vec_type.Mod |= TypeMod.V2;
                }
                else if (Elements.Count() == 4)
                {
                    if (vec_type != null) vec_type.Mod |= TypeMod.V4;
                }
                else
                {
                    vec_type = null;
                }

                return vec_type;
            }
        }

        protected override SoftwareIsa CustomVersion
        {
            get
            {
                var elt_version = ElementType.Version();
                var els_version = Elements.MaxOrDefault(el => el.Version());
                return (SoftwareIsa)Math.Max((int)elt_version, (int)els_version);
            }
        }

        protected override HardwareIsa CustomTarget
        {
            get
            {
                var elt_target = ElementType.Version();
                var els_target = Elements.MaxOrDefault(el => el.Target());
                return (HardwareIsa)Math.Max((int)elt_target, (int)els_target);
            }
        }

        protected override void CustomValidate()
        {
            (Type != null).AssertTrue();
            Type.Validate();

            (Elements != null).AssertTrue();
            Elements.ForEach(el =>
            {
                el.AssertNotNull();
                el.Validate();

                // todo. I've witnessed very strange behavior of ptxas:
                //
                // 1) I could use 0f00000000 in src vectors, tho cannot use 0, 
                // 2) I could use vars in src vectors, but in a strange way: 
                //   "mov.v4.b32 v4_u32, {a, foo, c, d}" works when foo is ".global .u32" 
                //   and ain't work when foo is ".global .f32".
                //
                // till this gets resolved (e.g. when I have more time to ask NVIDIA)
                // I only allow registers to comprise Vector literals

                (el is Reg).AssertTrue();
                agree(el, ElementType);
            });
        }

        protected override void RenderPtx()
        {
            writer.Write("{");

            Elements.ForEach((el, i) =>
            {
                if (i != 0) writer.Write(", ");
                el.RenderPtx();
            });

            writer.Write("}");
        }
    }

    [DebuggerNonUserCode]
    public static class VectorExtensions
    {
        public static ReadOnlyCollection<Expression> Flatten(this Vector vector)
        {
            return vector.Elements.AssertEach(el => el is Reg).ToReadOnly();
        }
    }
}