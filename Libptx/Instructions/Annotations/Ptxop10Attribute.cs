using System;
using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;

namespace Libptx.Instructions.Annotations
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
    [DebuggerNonUserCode]
    internal class Ptxop10Attribute : Atom10Attribute
    {
        public Ptxop10Attribute()
            : base()
        {
        }

        public Ptxop10Attribute(SoftwareIsa swisa)
            : base(swisa)
        {
        }

        public Ptxop10Attribute(HardwareIsa hwisa)
            : base(hwisa)
        {
        }

        public Ptxop10Attribute(SoftwareIsa swisa, HardwareIsa hwisa)
            : base(swisa, hwisa)
        {
        }

        public Ptxop10Attribute(String spec)
            : base(spec)
        {
        }

        public Ptxop10Attribute(String spec, SoftwareIsa swisa)
            : base(spec, swisa)
        {
        }

        public Ptxop10Attribute(String spec, HardwareIsa hwisa)
            : base(spec, hwisa)
        {
        }

        public Ptxop10Attribute(String spec, SoftwareIsa swisa, HardwareIsa hwisa)
            : base(spec, swisa, hwisa)
        {
        }
    }
}
