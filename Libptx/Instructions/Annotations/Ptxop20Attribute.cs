using System;
using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Atoms;

namespace Libptx.Instructions.Annotations
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
    [DebuggerNonUserCode]
    internal class Ptxop20Attribute : Atom20Attribute
    {
        public Ptxop20Attribute()
        {
        }

        public Ptxop20Attribute(SoftwareIsa swisa)
            : base(swisa)
        {
        }

        public Ptxop20Attribute(HardwareIsa hwisa)
            : base(hwisa)
        {
        }

        public Ptxop20Attribute(SoftwareIsa swisa, HardwareIsa hwisa)
            : base(swisa, hwisa)
        {
        }

        public Ptxop20Attribute(String spec)
            : base(spec)
        {
        }

        public Ptxop20Attribute(String spec, SoftwareIsa swisa)
            : base(spec, swisa)
        {
        }

        public Ptxop20Attribute(String spec, HardwareIsa hwisa)
            : base(spec, hwisa)
        {
        }

        public Ptxop20Attribute(String spec, SoftwareIsa swisa, HardwareIsa hwisa)
            : base(spec, swisa, hwisa)
        {
        }
    }
}
