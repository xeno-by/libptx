using System;
using System.Diagnostics;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Common.Annotations.Quantas
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, AllowMultiple = false, Inherited = false)]
    [DebuggerNonUserCode]
    internal class Suffix20Attribute : SuffixAttribute
    {
        public Suffix20Attribute()
            : this(null, SoftwareIsa.PTX_20, HardwareIsa.SM_20)
        {
        }

        public Suffix20Attribute(SoftwareIsa softwareIsa)
            : this(null, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), HardwareIsa.SM_20)
        {
        }

        public Suffix20Attribute(HardwareIsa hardwareIsa)
            : this(null, SoftwareIsa.PTX_20, hardwareIsa)
        {
        }

        public Suffix20Attribute(SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : this(null, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), hardwareIsa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }

        public Suffix20Attribute(HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : this(null, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), hardwareIsa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }

        public Suffix20Attribute(String signature)
            : this(signature, SoftwareIsa.PTX_20, HardwareIsa.SM_20)
        {
        }

        public Suffix20Attribute(String signature, SoftwareIsa softwareIsa)
            : this(signature, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), HardwareIsa.SM_10)
        {
        }

        public Suffix20Attribute(String signature, HardwareIsa hardwareIsa)
            : this(signature, SoftwareIsa.PTX_20, hardwareIsa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }

        public Suffix20Attribute(String signature, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : base(signature, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), hardwareIsa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }

        public Suffix20Attribute(String signature, HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : base(signature, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), hardwareIsa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }
    }
}