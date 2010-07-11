using System;
using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations.Core;

namespace Libptx.Instructions.Annotations
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
    [DebuggerNonUserCode]
    internal class PtxopAttribute : PtxopAnnotation
    {
        public PtxopAttribute()
            : base()
        {
        }

        public PtxopAttribute(SoftwareIsa swisa)
            : base(swisa)
        {
        }

        public PtxopAttribute(HardwareIsa hwisa)
            : base(hwisa)
        {
        }

        public PtxopAttribute(SoftwareIsa swisa, HardwareIsa hwisa)
            : base(swisa, hwisa)
        {
        }

        public PtxopAttribute(String spec)
            : base(spec)
        {
        }

        public PtxopAttribute(String spec, SoftwareIsa swisa)
            : base(spec, swisa)
        {
        }

        public PtxopAttribute(String spec, HardwareIsa hwisa)
            : base(spec, hwisa)
        {
        }

        public PtxopAttribute(String spec, SoftwareIsa swisa, HardwareIsa hwisa)
            : base(spec, swisa, hwisa)
        {
        }
    }
}
