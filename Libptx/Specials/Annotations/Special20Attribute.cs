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
    public class Special20Attribute : Atom20Attribute, SpecialAttribute
    {
        public virtual PtxType Type { get; set; }

        public Special20Attribute(Type type)
        {
            Type = type;
        }

        public Special20Attribute(Type type, SoftwareIsa swisa)
            : base(swisa)
        {
            Type = type;
        }

        public Special20Attribute(Type type, HardwareIsa hwisa)
            : base(hwisa)
        {
            Type = type;
        }

        public Special20Attribute(Type type, SoftwareIsa swisa, HardwareIsa hwisa)
            : base(swisa, hwisa)
        {
            Type = type;
        }

        public Special20Attribute(String spec, Type type)
            : base(spec)
        {
            Type = type;
        }

        public Special20Attribute(String spec, Type type, SoftwareIsa swisa)
            : base(spec, swisa)
        {
            Type = type;
        }

        public Special20Attribute(String spec, Type type, HardwareIsa hwisa)
            : base(spec, hwisa)
        {
            Type = type;
        }

        public Special20Attribute(String spec, Type type, SoftwareIsa swisa, HardwareIsa hwisa)
            : base(spec, swisa, hwisa)
        {
            Type = type;
        }
    }
}
