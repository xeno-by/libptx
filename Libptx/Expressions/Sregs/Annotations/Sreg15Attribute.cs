using System;
using System.Diagnostics;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Expressions.Sregs.Annotations
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = true)]
    [DebuggerNonUserCode]
    public class Sreg15Attribute : SregAttribute
    {
        public Sreg15Attribute(Type type)
            : this(null, type, SoftwareIsa.PTX_15, HardwareIsa.SM_10)
        {
        }

        public Sreg15Attribute(Type type, SoftwareIsa softwareIsa)
            : this(null, type, softwareIsa.AssertThat(isa => isa < SoftwareIsa.PTX_20), HardwareIsa.SM_10)
        {
        }

        public Sreg15Attribute(Type type, HardwareIsa hardwareIsa)
            : this(null, type, SoftwareIsa.PTX_15, hardwareIsa.AssertThat(isa => isa < HardwareIsa.SM_20))
        {
        }

        public Sreg15Attribute(Type type, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : this(null, type, softwareIsa, hardwareIsa)
        {
        }

        public Sreg15Attribute(Type type, HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : this(null, type, softwareIsa, hardwareIsa)
        {
        }

        public Sreg15Attribute(String signature, Type type)
            : this(signature, type, SoftwareIsa.PTX_15, HardwareIsa.SM_10)
        {
        }

        public Sreg15Attribute(String signature, Type type, SoftwareIsa softwareIsa)
            : this(signature, type, softwareIsa.AssertThat(isa => isa < SoftwareIsa.PTX_20), HardwareIsa.SM_10)
        {
        }

        public Sreg15Attribute(String signature, Type type, HardwareIsa hardwareIsa)
            : this(signature, type, SoftwareIsa.PTX_15, hardwareIsa.AssertThat(isa => isa < HardwareIsa.SM_20))
        {
        }

        public Sreg15Attribute(String signature, Type type, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : base(signature, type, softwareIsa, hardwareIsa)
        {
        }

        public Sreg15Attribute(String signature, Type type, HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : base(signature, type, softwareIsa, hardwareIsa)
        {
        }
    }
}