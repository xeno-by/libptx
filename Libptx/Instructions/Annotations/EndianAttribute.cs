using System;
using System.Diagnostics;
using Libcuda.Versions;

namespace Libptx.Instructions.Annotations
{
    [AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
    [DebuggerNonUserCode]
    internal class EndianAttribute : PtxopAttribute
    {
        public EndianAttribute()
        {
        }

        public EndianAttribute(SoftwareIsa swisa)
            : base(swisa)
        {
        }

        public EndianAttribute(HardwareIsa hwisa)
            : base(hwisa)
        {
        }

        public EndianAttribute(SoftwareIsa swisa, HardwareIsa hwisa)
            : base(swisa, hwisa)
        {
        }

        public EndianAttribute(String spec)
            : base(spec)
        {
        }

        public EndianAttribute(String spec, SoftwareIsa swisa)
            : base(spec, swisa)
        {
        }

        public EndianAttribute(String spec, HardwareIsa hwisa)
            : base(spec, hwisa)
        {
        }

        public EndianAttribute(String spec, SoftwareIsa swisa, HardwareIsa hwisa)
            : base(spec, swisa, hwisa)
        {
        }
    }
}