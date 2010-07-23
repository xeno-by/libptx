using System;
using System.Diagnostics;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Expressions.Specials.Annotations
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
    [DebuggerNonUserCode]
    public class Special20Attribute : SpecialAttribute
    {
        public Special20Attribute(Type type)
            : this(null, type, SoftwareIsa.PTX_20, HardwareIsa.SM_20)
        {
        }

        public Special20Attribute(Type type, SoftwareIsa softwareIsa)
            : this(null, type, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), HardwareIsa.SM_20)
        {
        }

        public Special20Attribute(Type type, HardwareIsa hardwareIsa)
            : this(null, type, SoftwareIsa.PTX_20, hardwareIsa)
        {
        }

        public Special20Attribute(Type type, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : this(null, type, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), hardwareIsa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }

        public Special20Attribute(Type type, HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : this(null, type, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), hardwareIsa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }

        public Special20Attribute(String signature, Type type)
            : this(signature, type, SoftwareIsa.PTX_20, HardwareIsa.SM_20)
        {
        }

        public Special20Attribute(String signature, Type type, SoftwareIsa softwareIsa)
            : this(signature, type, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), HardwareIsa.SM_20)
        {
        }

        public Special20Attribute(String signature, Type type, HardwareIsa hardwareIsa)
            : this(signature, type, SoftwareIsa.PTX_20, hardwareIsa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }

        public Special20Attribute(String signature, Type type, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : base(signature, type, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), hardwareIsa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }

        public Special20Attribute(String signature, Type type, HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : base(signature, type, softwareIsa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), hardwareIsa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }
    }
}