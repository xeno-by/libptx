using System;
using System.Diagnostics;
using Libcuda.Versions;

namespace Libptx.Common.Infrastructure
{
    [AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
    [DebuggerNonUserCode]
    internal class InfixAttribute : Atom10Attribute
    {
        public InfixAttribute()
            : base()
        {
        }

        public InfixAttribute(SoftwareIsa swisa)
            : base(swisa)
        {
        }

        public InfixAttribute(HardwareIsa hwisa)
            : base(hwisa)
        {
        }

        public InfixAttribute(SoftwareIsa swisa, HardwareIsa hwisa)
            : base(swisa, hwisa)
        {
        }

        public InfixAttribute(String signature)
            : base(signature)
        {
        }

        public InfixAttribute(String signature, SoftwareIsa swisa)
            : base(signature, swisa)
        {
        }

        public InfixAttribute(String signature, HardwareIsa hwisa)
            : base(signature, hwisa)
        {
        }

        public InfixAttribute(String signature, SoftwareIsa swisa, HardwareIsa hwisa)
            : base(signature, swisa, hwisa)
        {
        }
    }
}