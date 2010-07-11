using System;
using System.Diagnostics;
using Libcuda.Versions;

namespace Libptx.Instructions.Annotations
{
    [AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
    [DebuggerNonUserCode]
    internal class SuffixAttribute : PtxopAttribute
    {
        public SuffixAttribute()
        {
        }

        public SuffixAttribute(SoftwareIsa swisa)
            : base(swisa)
        {
        }

        public SuffixAttribute(HardwareIsa hwisa)
            : base(hwisa)
        {
        }

        public SuffixAttribute(SoftwareIsa swisa, HardwareIsa hwisa)
            : base(swisa, hwisa)
        {
        }

        public SuffixAttribute(String spec)
            : base(spec)
        {
        }

        public SuffixAttribute(String spec, SoftwareIsa swisa)
            : base(spec, swisa)
        {
        }

        public SuffixAttribute(String spec, HardwareIsa hwisa)
            : base(spec, hwisa)
        {
        }

        public SuffixAttribute(String spec, SoftwareIsa swisa, HardwareIsa hwisa)
            : base(spec, swisa, hwisa)
        {
        }
    }
}