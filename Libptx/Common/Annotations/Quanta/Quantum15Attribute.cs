using System;
using System.Diagnostics;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Common.Annotations.Quanta
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
    [DebuggerNonUserCode]
    public class Quantum15Attribute : ParticleAttribute
    {
        protected Quantum15Attribute()
            : this(null, SoftwareIsa.PTX_15, HardwareIsa.SM_10)
        {
        }

        protected Quantum15Attribute(SoftwareIsa softwareIsa)
            : this(null, softwareIsa.AssertThat(isa => isa < SoftwareIsa.PTX_20), HardwareIsa.SM_10)
        {
        }

        protected Quantum15Attribute(HardwareIsa hardwareIsa)
            : this(null, SoftwareIsa.PTX_15, hardwareIsa.AssertThat(isa => isa < HardwareIsa.SM_20))
        {
        }

        protected Quantum15Attribute(SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : this(null, softwareIsa, hardwareIsa)
        {
        }

        protected Quantum15Attribute(HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : this(null, softwareIsa, hardwareIsa)
        {
        }

        protected Quantum15Attribute(String signature)
            : this(signature, SoftwareIsa.PTX_15, HardwareIsa.SM_10)
        {
        }

        protected Quantum15Attribute(String signature, SoftwareIsa softwareIsa)
            : this(signature, softwareIsa.AssertThat(isa => isa < SoftwareIsa.PTX_20), HardwareIsa.SM_10)
        {
        }

        protected Quantum15Attribute(String signature, HardwareIsa hardwareIsa)
            : this(signature, SoftwareIsa.PTX_15, hardwareIsa.AssertThat(isa => isa < HardwareIsa.SM_20))
        {
        }

        protected Quantum15Attribute(String signature, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : base(signature, softwareIsa, hardwareIsa)
        {
        }

        protected Quantum15Attribute(String signature, HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : base(signature, softwareIsa, hardwareIsa)
        {
        }
    }
}