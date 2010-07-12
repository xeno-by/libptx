using System;
using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Type = System.Type;
using PtxType = Libptx.Common.Type;

namespace Libptx.Specials.Annotations
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
    [DebuggerNonUserCode]
    public class Special10Attribute : Atom10Attribute, SpecialAttribute
    {
        public virtual PtxType Type { get; set; }

        public Special10Attribute(Type type)
            : base()
        {
            Type = type;
        }

        public Special10Attribute(Type type, SoftwareIsa swisa)
            : base(swisa)
        {
            Type = type;
        }

        public Special10Attribute(Type type, HardwareIsa hwisa)
            : base(hwisa)
        {
            Type = type;
        }

        public Special10Attribute(Type type, SoftwareIsa swisa, HardwareIsa hwisa)
            : base(swisa, hwisa)
        {
            Type = type;
        }

        public Special10Attribute(String spec, Type type)
            : base(spec)
        {
            Type = type;
        }

        public Special10Attribute(String spec, Type type, SoftwareIsa swisa)
            : base(spec, swisa)
        {
            Type = type;
        }

        public Special10Attribute(String spec, Type type, HardwareIsa hwisa)
            : base(spec, hwisa)
        {
            Type = type;
        }

        public Special10Attribute(String spec, Type type, SoftwareIsa swisa, HardwareIsa hwisa)
            : base(spec, swisa, hwisa)
        {
            Type = type;
        }
    }
}
