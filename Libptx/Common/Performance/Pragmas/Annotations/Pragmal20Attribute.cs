using System;
using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Atoms;
using XenoGears.Assertions;

namespace Libptx.Common.Performance.Pragmas.Annotations
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
    [DebuggerNonUserCode]
    public class Pragma20Attribute : AtomAttribute
    {
        public Pragma20Attribute()
            : this(null, SoftwareIsa.PTX_20, HardwareIsa.SM_20)
        {
        }

        public Pragma20Attribute(SoftwareIsa softwareIsa)
            : this(null, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), HardwareIsa.SM_20)
        {
        }

        public Pragma20Attribute(HardwareIsa hardwareIsa)
            : this(null, SoftwareIsa.PTX_20, hardwareIsa)
        {
        }

        public Pragma20Attribute(SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : this(null, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), hardwareIsa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }

        public Pragma20Attribute(HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : this(null, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), hardwareIsa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }

        public Pragma20Attribute(String signature)
            : this(signature, SoftwareIsa.PTX_20, HardwareIsa.SM_20)
        {
        }

        public Pragma20Attribute(String signature, SoftwareIsa softwareIsa)
            : this(signature, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), HardwareIsa.SM_20)
        {
        }

        public Pragma20Attribute(String signature, HardwareIsa hardwareIsa)
            : this(signature, SoftwareIsa.PTX_20, hardwareIsa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }

        public Pragma20Attribute(String signature, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : base(signature, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), hardwareIsa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }

        public Pragma20Attribute(String signature, HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : base(signature, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), hardwareIsa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }
    }
}