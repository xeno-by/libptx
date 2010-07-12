using System;
using System.Diagnostics;
using Libcuda.Versions;

namespace Libptx.Common.Infrastructure
{
    [AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
    [DebuggerNonUserCode]
    internal class ModAttribute : Atom10Attribute
    {
        public ModAttribute()
            : base()
        {
        }

        public ModAttribute(SoftwareIsa swisa)
            : base(swisa)
        {
        }

        public ModAttribute(HardwareIsa hwisa)
            : base(hwisa)
        {
        }

        public ModAttribute(SoftwareIsa swisa, HardwareIsa hwisa)
            : base(swisa, hwisa)
        {
        }

        public ModAttribute(String signature)
            : base(signature)
        {
        }

        public ModAttribute(String signature, SoftwareIsa swisa)
            : base(signature, swisa)
        {
        }

        public ModAttribute(String signature, HardwareIsa hwisa)
            : base(signature, hwisa)
        {
        }

        public ModAttribute(String signature, SoftwareIsa swisa, HardwareIsa hwisa)
            : base(signature, swisa, hwisa)
        {
        }
    }
}